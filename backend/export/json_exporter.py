"""
JSON Exporter
Exports invoice data to JSON format with optional metadata
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class JSONExporter:
    """
    Exports invoice data to JSON format
    """
    
    def __init__(self):
        """Initialize JSON exporter"""
        pass
    
    def export(
        self,
        data: List[Dict],
        output_folder: str,
        filename: str = "invoice_data.json",
        include_metadata: bool = True,
        validation_issues: Dict = None,
        confidence_scores: Dict = None
    ) -> str:
        """
        Export data to JSON
        
        Args:
            data: List of invoice items
            output_folder: Output folder path
            filename: Output filename
            include_metadata: Include export metadata
            validation_issues: Validation issues
            confidence_scores: Confidence scores
            
        Returns:
            Path to exported file
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_path = os.path.join(output_folder, filename)
        
        # Build export structure
        export_data = {}
        
        if include_metadata:
            export_data['metadata'] = {
                'export_timestamp': datetime.now().isoformat(),
                'total_items': len(data),
                'format_version': '1.0'
            }
            
            if confidence_scores:
                export_data['metadata']['confidence'] = confidence_scores
        
        export_data['data'] = data
        
        if validation_issues:
            export_data['validation_issues'] = validation_issues
        
        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON file exported to: {output_path}")
        
        return output_path
    
    def export_simple(
        self,
        data: List[Dict],
        output_path: str,
        pretty: bool = True
    ) -> str:
        """
        Simple JSON export without metadata
        
        Args:
            data: List of invoice items
            output_path: Full output path
            pretty: Pretty print JSON (default: True)
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        print(f"JSON file saved at {output_path}")
        
        return output_path
    
    def export_structured(
        self,
        data: List[Dict],
        output_path: str,
        group_by_invoice: bool = True
    ) -> str:
        """
        Export JSON with structured grouping
        
        Args:
            data: List of invoice items
            output_path: Full output path
            group_by_invoice: Group items by invoice number
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if group_by_invoice:
            # Group by invoice number
            grouped = {}
            
            for item in data:
                inv_num = item.get('Invoice Number', 'Unknown')
                
                if inv_num not in grouped:
                    grouped[inv_num] = {
                        'invoice_number': inv_num,
                        'company_name': item.get('Company Name', 'N/A'),
                        'date': item.get('Date of Invoice', 'N/A'),
                        'items': []
                    }
                
                grouped[inv_num]['items'].append(item)
            
            export_data = {
                'invoices': list(grouped.values()),
                'total_invoices': len(grouped),
                'total_items': len(data)
            }
        else:
            export_data = {'items': data}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Structured JSON saved at {output_path}")
        
        return output_path