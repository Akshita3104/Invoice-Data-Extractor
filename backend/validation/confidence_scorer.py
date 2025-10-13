"""
Validation Confidence Scorer
Scores overall confidence of extracted and validated data
Combines multiple validation results into a single confidence score
"""

from typing import Dict, List
import numpy as np


class ValidationConfidenceScorer:
    """
    Calculates confidence scores for validated invoice data
    """
    
    def __init__(self):
        """Initialize confidence scorer"""
        # Weights for different validation aspects
        self.weights = {
            'arithmetic': 0.30,
            'format': 0.25,
            'consistency': 0.25,
            'plausibility': 0.20
        }
        
        # Severity penalties
        self.severity_penalties = {
            'critical': 0.50,
            'high': 0.30,
            'medium': 0.15,
            'low': 0.05,
            'info': 0.00
        }
    
    def calculate_confidence(
        self,
        extracted_data: List[Dict],
        arithmetic_issues: List[Dict],
        format_issues: List[Dict],
        consistency_issues: List[Dict],
        plausibility_issues: List[Dict]
    ) -> Dict:
        """
        Calculate overall confidence score
        
        Args:
            extracted_data: Extracted invoice data
            arithmetic_issues: Issues from arithmetic validation
            format_issues: Issues from format validation
            consistency_issues: Issues from consistency validation
            plausibility_issues: Issues from plausibility validation
            
        Returns:
            Dictionary with confidence scores and details
        """
        # Calculate individual confidence scores
        arithmetic_conf = self._calculate_validation_confidence(arithmetic_issues)
        format_conf = self._calculate_validation_confidence(format_issues)
        consistency_conf = self._calculate_validation_confidence(consistency_issues)
        plausibility_conf = self._calculate_validation_confidence(plausibility_issues)
        
        # Calculate weighted overall confidence
        overall_confidence = (
            arithmetic_conf * self.weights['arithmetic'] +
            format_conf * self.weights['format'] +
            consistency_conf * self.weights['consistency'] +
            plausibility_conf * self.weights['plausibility']
        )
        
        # Calculate completeness score
        completeness = self._calculate_completeness(extracted_data)
        
        # Final confidence (considering completeness)
        final_confidence = overall_confidence * 0.7 + completeness * 0.3
        
        # Determine quality level
        quality_level = self._get_quality_level(final_confidence)
        
        # Calculate field-level confidence
        field_confidence = self._calculate_field_confidence(extracted_data)
        
        return {
            'overall_confidence': round(final_confidence, 3),
            'quality_level': quality_level,
            'validation_scores': {
                'arithmetic': round(arithmetic_conf, 3),
                'format': round(format_conf, 3),
                'consistency': round(consistency_conf, 3),
                'plausibility': round(plausibility_conf, 3)
            },
            'completeness': round(completeness, 3),
            'field_confidence': field_confidence,
            'total_issues': {
                'arithmetic': len(arithmetic_issues),
                'format': len(format_issues),
                'consistency': len(consistency_issues),
                'plausibility': len(plausibility_issues)
            },
            'needs_review': final_confidence < 0.7,
            'ready_for_processing': final_confidence >= 0.8
        }
    
    def _calculate_validation_confidence(self, issues: List[Dict]) -> float:
        """
        Calculate confidence from validation issues
        
        Returns:
            Confidence score (0-1)
        """
        if not issues:
            return 1.0
        
        # Calculate penalty based on severity
        total_penalty = 0.0
        for issue in issues:
            severity = issue.get('severity', 'low')
            penalty = self.severity_penalties.get(severity, 0.10)
            total_penalty += penalty
        
        # Cap penalty at 1.0
        total_penalty = min(total_penalty, 1.0)
        
        # Confidence = 1 - penalty
        confidence = 1.0 - total_penalty
        
        return max(0.0, confidence)
    
    def _calculate_completeness(self, data: List[Dict]) -> float:
        """
        Calculate completeness score (how many required fields are present)
        """
        if not data:
            return 0.0
        
        # Required fields
        required_fields = [
            'Goods Description',
            'Quantity',
            'Rate',
            'Amount',
            'Company Name',
            'Invoice Number',
            'Date of Invoice'
        ]
        
        # Optional but important fields
        optional_fields = [
            'HSN/SAC Code',
            'Weight',
            'FSSAI Number'
        ]
        
        total_score = 0.0
        
        for item in data:
            item_score = 0.0
            
            # Check required fields (70% weight)
            present_required = sum(
                1 for field in required_fields 
                if item.get(field) and item.get(field) != 'N/A'
            )
            item_score += (present_required / len(required_fields)) * 0.7
            
            # Check optional fields (30% weight)
            present_optional = sum(
                1 for field in optional_fields
                if item.get(field) and item.get(field) != 'N/A'
            )
            item_score += (present_optional / len(optional_fields)) * 0.3
            
            total_score += item_score
        
        # Average across all items
        return total_score / len(data)
    
    def _calculate_field_confidence(self, data: List[Dict]) -> Dict:
        """
        Calculate confidence for individual fields
        """
        field_confidence = {}
        
        # Fields to check
        fields = [
            'Goods Description',
            'Quantity',
            'Rate',
            'Amount',
            'Company Name',
            'Invoice Number',
            'Date of Invoice',
            'HSN/SAC Code',
            'Weight'
        ]
        
        for field in fields:
            values = [item.get(field) for item in data]
            
            # Calculate confidence based on:
            # 1. Presence (not missing or N/A)
            # 2. Format validity
            # 3. Consistency across items
            
            present_count = sum(1 for v in values if v and v != 'N/A')
            presence_score = present_count / len(values) if values else 0.0
            
            # Format validity (simplified - would be more complex in production)
            valid_count = sum(1 for v in values if v and v != 'N/A' and len(str(v)) > 0)
            format_score = valid_count / len(values) if values else 0.0
            
            # Overall field confidence
            field_conf = (presence_score * 0.6 + format_score * 0.4)
            
            field_confidence[field] = round(field_conf, 3)
        
        return field_confidence
    
    def _get_quality_level(self, confidence: float) -> str:
        """
        Convert confidence score to quality level
        """
        if confidence >= 0.9:
            return 'excellent'
        elif confidence >= 0.8:
            return 'good'
        elif confidence >= 0.7:
            return 'acceptable'
        elif confidence >= 0.5:
            return 'poor'
        else:
            return 'very_poor'
    
    def score_item(
        self,
        item: Dict,
        validation_results: Dict = None
    ) -> float:
        """
        Calculate confidence score for a single item
        
        Args:
            item: Single invoice line item
            validation_results: Optional validation results for this item
            
        Returns:
            Confidence score (0-1)
        """
        score = 1.0
        
        # Check required fields
        required_fields = ['Goods Description', 'Quantity', 'Rate', 'Amount']
        for field in required_fields:
            if not item.get(field) or item.get(field) == 'N/A':
                score -= 0.15
        
        # Check if item has been corrected
        if item.get('_corrected', False):
            score -= 0.1
        
        # Apply validation penalties if provided
        if validation_results:
            issues = validation_results.get('issues', [])
            for issue in issues:
                severity = issue.get('severity', 'low')
                penalty = self.severity_penalties.get(severity, 0.05)
                score -= penalty
        
        return max(0.0, min(1.0, score))
    
    def recommend_actions(self, confidence_result: Dict) -> List[str]:
        """
        Recommend actions based on confidence scores
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        overall_conf = confidence_result['overall_confidence']
        
        if overall_conf < 0.5:
            recommendations.append('⚠️ CRITICAL: Manual review required - confidence very low')
            recommendations.append('Consider re-scanning the document with better quality')
        elif overall_conf < 0.7:
            recommendations.append('⚠️ Manual review recommended - several issues detected')
        
        # Check individual validation scores
        validation_scores = confidence_result['validation_scores']
        
        if validation_scores['arithmetic'] < 0.7:
            recommendations.append('Check arithmetic calculations (quantity × rate = amount)')
        
        if validation_scores['format'] < 0.7:
            recommendations.append('Verify date and code formats (HSN, FSSAI, Invoice Number)')
        
        if validation_scores['consistency'] < 0.7:
            recommendations.append('Check consistency across fields (company name, invoice number)')
        
        if validation_scores['plausibility'] < 0.7:
            recommendations.append('Verify if amounts and quantities seem reasonable')
        
        # Check completeness
        if confidence_result['completeness'] < 0.7:
            recommendations.append('Some required fields are missing - fill in manually')
        
        # Check field-level issues
        field_conf = confidence_result['field_confidence']
        low_conf_fields = [
            field for field, conf in field_conf.items() 
            if conf < 0.6
        ]
        
        if low_conf_fields:
            recommendations.append(f'Low confidence in fields: {", ".join(low_conf_fields)}')
        
        # No issues
        if not recommendations:
            recommendations.append('✓ All validations passed - data looks good!')
        
        return recommendations
    
    def generate_report(
        self,
        extracted_data: List[Dict],
        all_issues: Dict,
        confidence_result: Dict
    ) -> str:
        """
        Generate human-readable validation report
        
        Args:
            extracted_data: Extracted data
            all_issues: Dictionary with all validation issues
            confidence_result: Confidence calculation result
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall summary
        report.append("OVERALL CONFIDENCE")
        report.append("-" * 60)
        report.append(f"Confidence Score: {confidence_result['overall_confidence']:.1%}")
        report.append(f"Quality Level: {confidence_result['quality_level'].upper()}")
        report.append(f"Ready for Processing: {'YES ✓' if confidence_result['ready_for_processing'] else 'NO ✗'}")
        report.append("")
        
        # Validation breakdown
        report.append("VALIDATION BREAKDOWN")
        report.append("-" * 60)
        for val_type, score in confidence_result['validation_scores'].items():
            status = "✓" if score >= 0.8 else "✗"
            report.append(f"{val_type.capitalize():20} {score:.1%}  {status}")
        report.append(f"{'Completeness':20} {confidence_result['completeness']:.1%}")
        report.append("")
        
        # Issues summary
        report.append("ISSUES DETECTED")
        report.append("-" * 60)
        total_issues = confidence_result['total_issues']
        
        total = sum(total_issues.values())
        report.append(f"Total Issues: {total}")
        
        for val_type, count in total_issues.items():
            if count > 0:
                report.append(f"  - {val_type.capitalize()}: {count}")
        
        if total == 0:
            report.append("  No issues detected! ✓")
        
        report.append("")
        
        # Detailed issues (if any)
        if total > 0:
            report.append("DETAILED ISSUES")
            report.append("-" * 60)
            
            for val_type, issues in all_issues.items():
                if issues:
                    report.append(f"\n{val_type.upper()}:")
                    for i, issue in enumerate(issues[:5], 1):  # Show first 5
                        severity = issue.get('severity', 'unknown')
                        message = issue.get('message', 'Unknown issue')
                        report.append(f"  {i}. [{severity.upper()}] {message}")
                    
                    if len(issues) > 5:
                        report.append(f"  ... and {len(issues) - 5} more")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 60)
        recommendations = self.recommend_actions(confidence_result)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_confidence_metrics(self, confidence_result: Dict) -> Dict:
        """
        Export confidence metrics in structured format for storage/analysis
        """
        return {
            'timestamp': self._get_timestamp(),
            'overall_confidence': confidence_result['overall_confidence'],
            'quality_level': confidence_result['quality_level'],
            'validation_scores': confidence_result['validation_scores'],
            'completeness': confidence_result['completeness'],
            'total_issues': confidence_result['total_issues'],
            'needs_review': confidence_result['needs_review'],
            'ready_for_processing': confidence_result['ready_for_processing'],
            'field_confidence': confidence_result['field_confidence']
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def compare_extractions(
        self,
        extraction1: List[Dict],
        extraction2: List[Dict]
    ) -> float:
        """
        Compare two extractions and return similarity score
        (Useful for comparing multiple OCR engine outputs)
        
        Returns:
            Similarity score (0-1)
        """
        if not extraction1 or not extraction2:
            return 0.0
        
        # Compare key fields
        fields_to_compare = ['Invoice Number', 'Company Name', 'Date of Invoice', 'Amount']
        
        matches = 0
        total = 0
        
        for field in fields_to_compare:
            values1 = [item.get(field) for item in extraction1]
            values2 = [item.get(field) for item in extraction2]
            
            # Compare each position
            for v1, v2 in zip(values1, values2):
                total += 1
                if v1 and v2 and str(v1).strip() == str(v2).strip():
                    matches += 1
        
        return matches / total if total > 0 else 0.0