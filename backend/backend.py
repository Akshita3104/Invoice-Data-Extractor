"""
Complete Invoice Extraction Pipeline
Integrates all modules into a unified extraction system
"""

import os
import sys
import json
from typing import Dict, List, Optional

# Import all modules
from ingestion import FormatHandler, QualityAssessor
from preprocessing import QualityEnhancer
from ocr import OCRRouter
from layout_analysis import ZoneSegmenter, TableDetector, ReadingOrderDetector
from graph import GraphBuilder, GNNReasoner
from multimodal import MultimodalFeatureExtractor, FusionLayer
from extraction import HybridExtractor
from validation import (
    ArithmeticValidator,
    FormatValidator,
    ConsistencyValidator,
    PlausibilityValidator,
    ValidationConfidenceScorer
)
from export import ExcelExporter, CSVExporter, JSONExporter


class InvoiceExtractionPipeline:
    """
    Complete invoice extraction pipeline integrating all modules
    """
    
    def __init__(
        self,
        api_key: str,
        enable_advanced_features: bool = True,
        use_gpu: bool = True
    ):
        """
        Initialize complete pipeline
        
        Args:
            api_key: LLM API key (Gemini)
            enable_advanced_features: Enable graph, multimodal fusion
            use_gpu: Use GPU for deep learning models
        """
        self.api_key = api_key
        self.enable_advanced = enable_advanced_features
        self.use_gpu = use_gpu
        
        print("Initializing Invoice Extraction Pipeline...")
        
        # Initialize all components
        self._initialize_components()
        
        print("Pipeline initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # Ingestion
        self.format_handler = FormatHandler(target_dpi=300)
        self.quality_assessor = QualityAssessor()
        
        # Preprocessing
        self.quality_enhancer = QualityEnhancer(aggressive=False)
        
        # OCR
        self.ocr_router = OCRRouter(enable_ensemble=False)
        
        # Layout Analysis
        self.zone_segmenter = ZoneSegmenter(method='hybrid')
        self.table_detector = TableDetector(method='hybrid')
        self.reading_order = ReadingOrderDetector()
        
        # Graph (if enabled)
        if self.enable_advanced:
            self.graph_builder = GraphBuilder(
                include_words=True,
                include_lines=True,
                include_blocks=True
            )
            self.gnn_reasoner = GNNReasoner(use_gpu=self.use_gpu)
        
        # Multimodal (if enabled)
        if self.enable_advanced:
            self.feature_extractor = MultimodalFeatureExtractor(
                use_visual=True,
                use_text=True,
                use_layout=True,
                use_graph=True,
                use_gpu=self.use_gpu
            )
            self.fusion_layer = FusionLayer(fusion_method='attention')
        
        # Extraction
        self.extractor = HybridExtractor(api_key=self.api_key, prefer_llm=True)
        
        # Validation
        self.arithmetic_validator = ArithmeticValidator(tolerance=0.02)
        self.format_validator = FormatValidator()
        self.consistency_validator = ConsistencyValidator()
        self.plausibility_validator = PlausibilityValidator()
        self.confidence_scorer = ValidationConfidenceScorer()
        
        # Export
        self.excel_exporter = ExcelExporter()
        self.csv_exporter = CSVExporter()
        self.json_exporter = JSONExporter()
    
    def process_document(
        self,
        document_path: str,
        output_folder: str,
        filename: str = "invoice_data.xlsx"
    ) -> Dict:
        """
        Process a single document through complete pipeline
        
        Args:
            document_path: Path to document (PDF, JPEG, PNG, TIFF)
            output_folder: Output folder for results
            filename: Output filename
            
        Returns:
            Processing results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(document_path)}")
        print(f"{'='*60}\n")
        
        results = {
            'input_file': document_path,
            'success': False,
            'steps': {},
            'extracted_data': None,
            'validation_issues': {},
            'confidence': None,
            'output_files': []
        }
        
        try:
            # Step 1: Ingestion
            print("Step 1: Document Ingestion...")
            image, quality_metrics = self.format_handler.process(document_path)
            results['steps']['ingestion'] = {
                'quality_score': quality_metrics['overall_score'],
                'quality_level': quality_metrics['quality_level']
            }
            print(f"  Quality: {quality_metrics['quality_level']} ({quality_metrics['overall_score']:.2f})")
            
            # Step 2: Preprocessing (if needed)
            if quality_metrics['overall_score'] < 0.7:
                print("\nStep 2: Adaptive Preprocessing...")
                image = self.quality_enhancer.enhance(image, quality_metrics)
                results['steps']['preprocessing'] = 'applied'
            else:
                print("\nStep 2: Preprocessing skipped (good quality)")
                results['steps']['preprocessing'] = 'skipped'
            
            # Step 3: OCR
            print("\nStep 3: OCR Extraction...")
            ocr_result = self.ocr_router.extract_text(image, quality_metrics)
            results['steps']['ocr'] = {
                'engine': ocr_result['engine'],
                'confidence': ocr_result['confidence']
            }
            print(f"  Engine: {ocr_result['engine']}, Confidence: {ocr_result['confidence']:.2f}")
            
            # Step 4: Layout Analysis
            print("\nStep 4: Layout Analysis...")
            zones = self.zone_segmenter.segment(image, ocr_result)
            tables = self.table_detector.detect(image, zones, ocr_result)
            reading_order_blocks = self.reading_order.detect_order(ocr_result, zones)
            
            results['steps']['layout'] = {
                'zones': len(zones),
                'tables': len(tables)
            }
            print(f"  Zones: {len(zones)}, Tables: {len(tables)}")
            
            # Step 5: Graph Construction (if enabled)
            if self.enable_advanced:
                print("\nStep 5: Document Graph Construction...")
                doc_graph = self.graph_builder.build(ocr_result, zones, tables)
                results['steps']['graph'] = {
                    'nodes': len(doc_graph.nodes()),
                    'edges': len(doc_graph.edges())
                }
                print(f"  Nodes: {len(doc_graph.nodes())}, Edges: {len(doc_graph.edges())}")
            else:
                doc_graph = None
            
            # Step 6: Multimodal Feature Extraction (if enabled)
            if self.enable_advanced:
                print("\nStep 6: Multimodal Feature Extraction...")
                features = self.feature_extractor.extract_features(
                    image=image,
                    ocr_result=ocr_result,
                    zones=zones,
                    graph=doc_graph
                )
                fused_features = self.fusion_layer.fuse(features)
                results['steps']['multimodal'] = 'completed'
            
            # Step 7: Extraction
            print("\nStep 7: Data Extraction...")
            extracted_data = self.extractor.extract(
                text=ocr_result['text'],
                ocr_result=ocr_result,
                zones=zones,
                tables=tables,
                graph_features=results['steps'].get('graph')
            )
            results['extracted_data'] = extracted_data
            results['steps']['extraction'] = {
                'items_extracted': len(extracted_data)
            }
            print(f"  Extracted: {len(extracted_data)} items")
            
            # Step 8: Validation
            print("\nStep 8: Multi-Layer Validation...")
            data, arith_issues = self.arithmetic_validator.validate(extracted_data)
            data, format_issues = self.format_validator.validate(data)
            data, consist_issues = self.consistency_validator.validate(data)
            data, plaus_issues = self.plausibility_validator.validate(data)
            
            results['validation_issues'] = {
                'arithmetic': arith_issues,
                'format': format_issues,
                'consistency': consist_issues,
                'plausibility': plaus_issues
            }
            
            total_issues = sum(len(issues) for issues in results['validation_issues'].values())
            print(f"  Total issues found: {total_issues}")
            
            # Step 9: Confidence Scoring
            print("\nStep 9: Confidence Scoring...")
            confidence_result = self.confidence_scorer.calculate_confidence(
                data,
                arith_issues,
                format_issues,
                consist_issues,
                plaus_issues
            )
            results['confidence'] = confidence_result
            print(f"  Overall Confidence: {confidence_result['overall_confidence']:.1%}")
            print(f"  Quality Level: {confidence_result['quality_level'].upper()}")
            
            # Step 10: Export
            print("\nStep 10: Exporting Results...")
            
            # Excel export
            excel_path = self.excel_exporter.export(
                data=data,
                output_folder=output_folder,
                filename=filename,
                include_validation=True,
                validation_issues=results['validation_issues'],
                confidence_scores=confidence_result
            )
            results['output_files'].append(excel_path)
            print(f"  Excel: {excel_path}")
            
            # JSON export (optional)
            json_filename = filename.replace('.xlsx', '.json')
            json_path = self.json_exporter.export(
                data=data,
                output_folder=output_folder,
                filename=json_filename,
                include_metadata=True,
                validation_issues=results['validation_issues'],
                confidence_scores=confidence_result
            )
            results['output_files'].append(json_path)
            print(f"  JSON: {json_path}")
            
            results['success'] = True
            
            print(f"\n{'='*60}")
            print("Processing Complete!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        return results
    
    def process_multiple_documents(
        self,
        document_paths: List[str],
        output_folder: str,
        filename: str = "combined_invoice_data.xlsx"
    ) -> Dict:
        """
        Process multiple documents
        
        Args:
            document_paths: List of document paths
            output_folder: Output folder
            filename: Output filename
            
        Returns:
            Combined processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing {len(document_paths)} documents")
        print(f"{'='*60}\n")
        
        all_data = []
        all_results = []
        combined_issues = {
            'arithmetic': [],
            'format': [],
            'consistency': [],
            'plausibility': []
        }
        
        for i, doc_path in enumerate(document_paths, 1):
            print(f"\n[{i}/{len(document_paths)}] Processing: {os.path.basename(doc_path)}")
            
            result = self.process_document(
                doc_path,
                output_folder,
                f"temp_{i}_{filename}"
            )
            
            if result['success'] and result['extracted_data']:
                all_data.extend(result['extracted_data'])
                all_results.append(result)
                
                # Combine issues
                for issue_type in combined_issues:
                    combined_issues[issue_type].extend(
                        result['validation_issues'].get(issue_type, [])
                    )
        
        # Export combined data
        if all_data:
            print(f"\n{'='*60}")
            print("Exporting Combined Results...")
            print(f"{'='*60}\n")
            
            # Calculate combined confidence
            confidence_result = self.confidence_scorer.calculate_confidence(
                all_data,
                combined_issues['arithmetic'],
                combined_issues['format'],
                combined_issues['consistency'],
                combined_issues['plausibility']
            )
            
            # Export
            excel_path = self.excel_exporter.export(
                data=all_data,
                output_folder=output_folder,
                filename=filename,
                include_validation=True,
                validation_issues=combined_issues,
                confidence_scores=confidence_result
            )
            
            print(f"Combined Excel: {excel_path}")
            print(f"Total items extracted: {len(all_data)}")
            print(f"Overall confidence: {confidence_result['overall_confidence']:.1%}")
        
        return {
            'total_documents': len(document_paths),
            'successful': len(all_results),
            'total_items': len(all_data),
            'combined_output': excel_path if all_data else None,
            'individual_results': all_results
        }


# ===================== Main Execution (For Electron) =====================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: backend.py <pdf_path1> <pdf_path2> ... <api_key> <output_folder> <filename>")
        sys.exit(1)
    
    # Parse arguments
    pdf_paths = sys.argv[1:-3]  # All PDF paths
    api_key = sys.argv[-3]
    output_folder = sys.argv[-2]
    filename = sys.argv[-1]
    
    try:
        # Initialize pipeline
        pipeline = InvoiceExtractionPipeline(
            api_key=api_key,
            enable_advanced_features=False,  # Set to True for full features
            use_gpu=False  # Set to True if GPU available
        )
        
        # Process documents
        if len(pdf_paths) == 1:
            # Single document
            result = pipeline.process_document(pdf_paths[0], output_folder, filename)
            
            if result['success']:
                print(f"\n✅ Success! Output: {result['output_files'][0]}")
            else:
                print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
        else:
            # Multiple documents
            result = pipeline.process_multiple_documents(pdf_paths, output_folder, filename)
            
            if result['combined_output']:
                print(f"\n✅ Success! Combined output: {result['combined_output']}")
            else:
                print(f"\n❌ Processing failed")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)