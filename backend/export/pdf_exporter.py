"""
PDF Report Exporter
Generates PDF reports with invoice data and validation summary
"""

import os
from typing import Dict, List, Optional
from datetime import datetime


class PDFReportExporter:
    """
    Exports invoice data as PDF report
    """
    
    def __init__(self):
        """Initialize PDF exporter"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            self.has_reportlab = True
            self.colors = colors
            self.letter = letter
            self.A4 = A4
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Table = Table
            self.TableStyle = TableStyle
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.PageBreak = PageBreak
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.inch = inch
            
        except ImportError:
            self.has_reportlab = False
            print("Warning: reportlab not available. Install with: pip install reportlab")
    
    def export(
        self,
        data: List[Dict],
        output_folder: str,
        filename: str = "invoice_report.pdf",
        validation_issues: Dict = None,
        confidence_scores: Dict = None
    ) -> Optional[str]:
        """
        Export data as PDF report
        
        Args:
            data: List of invoice items
            output_folder: Output folder path
            filename: Output filename
            validation_issues: Validation issues
            confidence_scores: Confidence scores
            
        Returns:
            Path to exported file or None if reportlab not available
        """
        if not self.has_reportlab:
            print("Cannot export PDF: reportlab not installed")
            return None
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Ensure filename has .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        output_path = os.path.join(output_folder, filename)
        
        # Create PDF
        doc = self.SimpleDocTemplate(output_path, pagesize=self.A4)
        story = []
        
        # Styles
        styles = self.getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Title
        story.append(self.Paragraph("Invoice Extraction Report", title_style))
        story.append(self.Spacer(1, 0.3 * self.inch))
        
        # Summary section
        story.append(self.Paragraph("Summary", heading_style))
        story.append(self.Spacer(1, 0.1 * self.inch))
        
        summary_data = [
            ['Total Items:', str(len(data))],
            ['Export Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        if confidence_scores:
            summary_data.append([
                'Overall Confidence:',
                f"{confidence_scores.get('overall_confidence', 0):.1%}"
            ])
            summary_data.append([
                'Quality Level:',
                confidence_scores.get('quality_level', 'unknown').upper()
            ])
        
        summary_table = self.Table(summary_data, colWidths=[2*self.inch, 3*self.inch])
        summary_table.setStyle(self.TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, self.colors.black)
        ]))
        
        story.append(summary_table)
        story.append(self.Spacer(1, 0.3 * self.inch))
        
        # Validation section (if issues present)
        if validation_issues:
            total_issues = sum(len(issues) for issues in validation_issues.values())
            
            story.append(self.Paragraph(f"Validation Issues ({total_issues} found)", heading_style))
            story.append(self.Spacer(1, 0.1 * self.inch))
            
            for validator_type, issues in validation_issues.items():
                if issues:
                    story.append(self.Paragraph(
                        f"<b>{validator_type.capitalize()}:</b> {len(issues)} issue(s)",
                        normal_style
                    ))
                    
                    # Show first 5 issues
                    for issue in issues[:5]:
                        message = issue.get('message', 'Unknown issue')
                        severity = issue.get('severity', 'unknown')
                        story.append(self.Paragraph(
                            f"  • [{severity.upper()}] {message}",
                            normal_style
                        ))
                    
                    if len(issues) > 5:
                        story.append(self.Paragraph(
                            f"  ... and {len(issues) - 5} more",
                            normal_style
                        ))
                    
                    story.append(self.Spacer(1, 0.1 * self.inch))
        
        # Data table
        story.append(self.PageBreak())
        story.append(self.Paragraph("Invoice Data", heading_style))
        story.append(self.Spacer(1, 0.1 * self.inch))
        
        if data:
            # Create table data
            headers = ['Description', 'Quantity', 'Rate', 'Amount']
            table_data = [headers]
            
            for item in data[:20]:  # Limit to 20 items
                row = [
                    str(item.get('Goods Description', 'N/A'))[:30],
                    str(item.get('Quantity', 'N/A')),
                    str(item.get('Rate', 'N/A'))[:20],
                    str(item.get('Amount', 'N/A'))
                ]
                table_data.append(row)
            
            if len(data) > 20:
                table_data.append(['...', '...', '...', '...'])
                table_data.append([f"Total: {len(data)} items", '', '', ''])
            
            data_table = self.Table(table_data)
            data_table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), self.colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black)
            ]))
            
            story.append(data_table)
        else:
            story.append(self.Paragraph("No data to display", normal_style))
        
        # Build PDF
        doc.build(story)
        
        print(f"PDF report exported to: {output_path}")
        
        return output_path
    
    def export_simple(
        self,
        data: List[Dict],
        output_path: str,
        title: str = "Invoice Report"
    ) -> Optional[str]:
        """
        Simple PDF export without validation
        
        Args:
            data: List of invoice items
            output_path: Full output path
            title: Report title
            
        Returns:
            Path to exported file or None
        """
        if not self.has_reportlab:
            print("Cannot export PDF: reportlab not installed")
            return None
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = self.SimpleDocTemplate(output_path, pagesize=self.A4)
        story = []
        
        styles = self.getSampleStyleSheet()
        
        # Title
        story.append(self.Paragraph(title, styles['Heading1']))
        story.append(self.Spacer(1, 0.3 * self.inch))
        
        # Data table
        if data:
            headers = list(data[0].keys())[:6]  # First 6 columns
            table_data = [headers]
            
            for item in data:
                row = [str(item.get(h, 'N/A'))[:30] for h in headers]
                table_data.append(row)
            
            data_table = self.Table(table_data)
            data_table.setStyle(self.TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, self.colors.black)
            ]))
            
            story.append(data_table)
        
        doc.build(story)
        
        print(f"PDF saved at {output_path}")
        
        return output_path