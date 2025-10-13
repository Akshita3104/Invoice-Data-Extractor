"""
Gemini Extractor
Extracts structured invoice data using Google's Gemini LLM
Enhanced version of the original extraction logic
"""

import json
import re
from typing import Dict, List, Optional
import google.generativeai as genai


class GeminiExtractor:
    """
    Extracts structured data from invoices using Gemini LLM
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini extractor
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Gemini extractor initialized with {model_name}")
    
    def extract(
        self,
        text: str,
        image: Optional[bytes] = None,
        extraction_schema: Dict = None
    ) -> List[Dict]:
        """
        Extract structured data from text/image
        
        Args:
            text: OCR extracted text
            image: Optional image bytes for vision models
            extraction_schema: Optional custom schema
            
        Returns:
            List of extracted invoice items
        """
        # Build prompt
        prompt = self._build_extraction_prompt(text, extraction_schema)
        
        try:
            # Generate response
            if image:
                # Use vision model
                response = self.model.generate_content([prompt, image])
            else:
                # Use text-only model
                response = self.model.generate_content(prompt)
            
            # Parse response
            extracted_data = self._parse_response(response.text)
            
            return extracted_data
            
        except Exception as e:
            print(f"Error in Gemini extraction: {e}")
            return []
    
    def _build_extraction_prompt(
        self,
        text: str,
        schema: Optional[Dict] = None
    ) -> str:
        """
        Build extraction prompt for Gemini
        """
        if schema:
            # Use custom schema
            fields = schema.get('fields', [])
            field_descriptions = '\n'.join([
                f"- {field['name']}: {field.get('description', '')}"
                for field in fields
            ])
        else:
            # Use default invoice schema
            field_descriptions = """
- Goods Description: The name or description of the product/item
- HSN/SAC Code: The HSN (for goods) or SAC (for services) code
- Quantity: The quantity of the product (numerical value only)
- Weight: The weight with unit (e.g., "50 KG", "0.5 MT", "10 Quintal")
- Rate: The rate per unit with currency (e.g., "₹2200/MT", "Rs.50 per kg")
- Amount: The total amount for this line item (not the invoice total)
- Company Name: The name of the company issuing the invoice
- Invoice Number: The invoice number (alphanumeric code)
- FSSAI Number: The FSSAI license number (if applicable, take buyer's FSSAI if two present)
- Date of Invoice: The invoice date in DD/MM/YYYY format
"""
        
        prompt = f"""
You are an expert in extracting structured data from invoices. Extract the following details from the text below:

{field_descriptions}

**IMPORTANT RULES**:
1. If a field is missing or unclear, set it to "N/A"
2. Do NOT infer or guess values - extract only what is explicitly present
3. For dates, convert to DD/MM/YYYY format
4. For quantities, extract only the numerical value
5. For rates, include both the amount and unit (e.g., "₹50/kg")
6. For amounts, extract the line item amount, NOT the total invoice amount
7. If multiple products are listed, extract each one separately
8. Keep the exact wording from the document
9. For weight, always include the unit (KG, MT, Qtl, etc.)
10. If FSSAI number appears twice, take the buyer's FSSAI number

**TEXT TO ANALYZE**:
{text}

**OUTPUT FORMAT**:
Return the result as a JSON array containing objects for each line item. Each object should have the fields listed above.
Example format:
[
  {{
    "Goods Description": "Rice",
    "HSN/SAC Code": "1006",
    "Quantity": "100",
    "Weight": "100 KG",
    "Rate": "₹50/kg",
    "Amount": "₹5000",
    "Company Name": "ABC Traders Pvt Ltd",
    "Invoice Number": "INV-2024-001",
    "FSSAI Number": "12345678901234",
    "Date of Invoice": "15/01/2024"
  }}
]

Return ONLY the JSON array, no other text or markdown formatting.
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> List[Dict]:
        """
        Parse Gemini response into structured data
        """
        try:
            # Clean response (remove markdown formatting)
            cleaned = response_text.strip()
            cleaned = cleaned.strip('`')
            
            # Remove "json" prefix if present
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            
            # Parse JSON
            extracted_data = json.loads(cleaned)
            
            # Ensure it's a list
            if isinstance(extracted_data, dict):
                extracted_data = [extracted_data]
            
            # Post-process extracted data
            extracted_data = self._post_process_data(extracted_data)
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}")
            
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                    return self._post_process_data(extracted_data)
                except:
                    pass
            
            return []
    
    def _post_process_data(self, data: List[Dict]) -> List[Dict]:
        """
        Post-process extracted data for consistency
        """
        processed = []
        
        for item in data:
            # Standardize field names
            processed_item = {}
            
            # Map various field names to standard names
            field_mapping = {
                'goods_description': 'Goods Description',
                'description': 'Goods Description',
                'item': 'Goods Description',
                'hsn_sac_code': 'HSN/SAC Code',
                'hsn': 'HSN/SAC Code',
                'sac': 'HSN/SAC Code',
                'quantity': 'Quantity',
                'qty': 'Quantity',
                'weight': 'Weight',
                'rate': 'Rate',
                'price': 'Rate',
                'amount': 'Amount',
                'total': 'Amount',
                'company_name': 'Company Name',
                'company': 'Company Name',
                'invoice_number': 'Invoice Number',
                'invoice_no': 'Invoice Number',
                'fssai_number': 'FSSAI Number',
                'fssai': 'FSSAI Number',
                'date_of_invoice': 'Date of Invoice',
                'date': 'Date of Invoice',
                'invoice_date': 'Date of Invoice'
            }
            
            for key, value in item.items():
                # Normalize key
                normalized_key = field_mapping.get(key.lower().replace(' ', '_'), key)
                processed_item[normalized_key] = value
            
            # Clean values
            for key in processed_item:
                if isinstance(processed_item[key], str):
                    processed_item[key] = processed_item[key].strip()
            
            processed.append(processed_item)
        
        return processed
    
    def extract_with_context(
        self,
        text: str,
        zones: List = None,
        tables: List = None,
        graph_features: Dict = None
    ) -> List[Dict]:
        """
        Extract with additional context from layout analysis
        
        Args:
            text: OCR text
            zones: Layout zones
            tables: Detected tables
            graph_features: Features from document graph
            
        Returns:
            Extracted data
        """
        # Build enhanced prompt with context
        context_info = []
        
        if zones:
            context_info.append(f"The document has {len(zones)} zones (header, body, footer)")
        
        if tables:
            context_info.append(f"The document contains {len(tables)} table(s)")
        
        if graph_features:
            num_nodes = graph_features.get('num_nodes', 0)
            context_info.append(f"Document structure has {num_nodes} text elements")
        
        # Add context to text
        if context_info:
            enhanced_text = f"CONTEXT: {'. '.join(context_info)}\n\n{text}"
        else:
            enhanced_text = text
        
        return self.extract(enhanced_text)
    
    def extract_specific_fields(
        self,
        text: str,
        fields: List[str]
    ) -> Dict:
        """
        Extract only specific fields
        
        Args:
            text: OCR text
            fields: List of field names to extract
            
        Returns:
            Dictionary with requested fields
        """
        # Build targeted prompt
        field_list = '\n'.join([f"- {field}" for field in fields])
        
        prompt = f"""
Extract only the following fields from the invoice text:

{field_list}

Text:
{text}

Return as JSON object with these fields. If a field is not found, use "N/A".
"""
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text)
            
            if result and isinstance(result, list):
                return result[0]
            return {}
            
        except Exception as e:
            print(f"Error extracting specific fields: {e}")
            return {}
    
    def validate_extraction(
        self,
        extracted_data: List[Dict],
        original_text: str
    ) -> List[Dict]:
        """
        Validate extracted data against original text
        """
        validated_data = []
        
        for item in extracted_data:
            # Check if key values exist in original text
            description = item.get('Goods Description', '')
            amount = item.get('Amount', '')
            
            if description and description != 'N/A':
                # Check if description appears in text
                if description.lower() not in original_text.lower():
                    item['_validation_warning'] = 'Description not found in original text'
            
            validated_data.append(item)
        
        return validated_data
    
    def batch_extract(
        self,
        texts: List[str]
    ) -> List[List[Dict]]:
        """
        Extract from multiple documents
        
        Args:
            texts: List of OCR texts
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"Processing document {i+1}/{len(texts)}...")
            extracted = self.extract(text)
            results.append(extracted)
        
        return results
    