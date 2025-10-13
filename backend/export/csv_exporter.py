"""
CSV Exporter
Exports invoice data to CSV format
"""

import os
import csv
from typing import Dict, List


class CSVExporter:
    """
    Exports invoice data to CSV format
    """
    
    def __init__(self):
        """Initialize CSV exporter"""
        self.default_columns = [
            'Goods Description',
            'HSN/SAC Code',
            'Quantity',
            'Weight',
            'Rate',
            'Amount',
            'Company Name',
            'Invoice Number',
            'FSSAI Number',
            'Date of Invoice'
        ]
    
    def export(
        self,
        data: List[Dict],
        output_folder: str,
        filename: str = "invoice_data.csv"
    ) -> str:
        """
        Export data to CSV
        
        Args:
            data: List of invoice items
            output_folder: Output folder path
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        output_path = os.path.join(output_folder, filename)
        
        if not data:
            # Write empty CSV with headers
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.default_columns)
                writer.writeheader()
        else:
            # Get all unique field names
            all_fields = set()
            for item in data:
                all_fields.update(item.keys())
            
            # Prioritize default columns
            fieldnames = [col for col in self.default_columns if col in all_fields]
            other_fields = [col for col in all_fields if col not in self.default_columns]
            fieldnames.extend(sorted(other_fields))
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in data:
                    # Ensure all fields exist
                    row = {field: item.get(field, 'N/A') for field in fieldnames}
                    writer.writerow(row)
        
        print(f"CSV file exported to: {output_path}")
        
        return output_path
    
    def export_with_delimiter(
        self,
        data: List[Dict],
        output_path: str,
        delimiter: str = ','
    ) -> str:
        """
        Export CSV with custom delimiter
        
        Args:
            data: List of invoice items
            output_path: Full output path
            delimiter: CSV delimiter (default: comma)
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not data:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.default_columns, delimiter=delimiter)
                writer.writeheader()
        else:
            fieldnames = list(data[0].keys())
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
        
        print(f"CSV file saved at {output_path}")
        
        return output_path