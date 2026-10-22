# search_functionality.py
# Script to implement search functions for medicines and drugs

import pandas as pd
import os
import re
from fuzzywuzzy import process, fuzz  # For fuzzy matching

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DrugSearchEngine:
    """
    A search engine for medicine and drug information
    """
    def __init__(self, 
                 medicine_data_path=None,
                 drug_properties_path=None,
                 repurposing_data_path=None):
        """
        Initialize the search engine with the necessary datasets
        """
        medicine_data_path = medicine_data_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv')
        drug_properties_path = drug_properties_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv')
        repurposing_data_path = repurposing_data_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'cleaned_repurposing_data.csv')

        # Load datasets
        self.med_data = pd.read_csv(medicine_data_path) if os.path.exists(medicine_data_path) else None
        self.drug_props = pd.read_csv(drug_properties_path) if os.path.exists(drug_properties_path) else None
        self.rep_data = pd.read_csv(repurposing_data_path) if os.path.exists(repurposing_data_path) else None
        
        # Create lookup dictionaries for faster access
        self.medicine_dict = {}
        self.drug_dict = {}
        self.medicine_to_drugs = {}
        self.drug_to_medicines = {}
        
        if self.med_data is not None:
            # Prepare medicine lookup
            for idx, row in self.med_data.iterrows():
                med_name = row['medicine_name'].lower() if 'medicine_name' in row else None
                if med_name:
                    self.medicine_dict[med_name] = row.to_dict()
                    
                    # Extract drugs from composition
                    if 'composition' in row and not pd.isna(row['composition']):
                        drugs = self._extract_drugs_from_composition(row['composition'])
                        self.medicine_to_drugs[med_name] = drugs
                        
                        # Map drugs back to medicine
                        for drug in drugs:
                            if drug not in self.drug_to_medicines:
                                self.drug_to_medicines[drug] = []
                            self.drug_to_medicines[drug].append(med_name)
        
        if self.drug_props is not None:
            # Prepare drug lookup
            for idx, row in self.drug_props.iterrows():
                drug_name = row['drug'].lower() if 'drug' in row else None
                if drug_name:
                    self.drug_dict[drug_name] = row.to_dict()
    
    def _extract_drugs_from_composition(self, composition):
        """
        Extract individual drugs from composition string
        """
        if pd.isna(composition):
            return []
        
        # Split by common separators
        drugs = []
        for sep in [',', '+', 'and', ';']:
            if sep in composition:
                drugs.extend([drug.strip().lower() for drug in composition.split(sep)])
                return drugs
        
        # If no separators found, assume it's a single drug
        return [composition.strip().lower()]
    
    def search_medicine(self, query, fuzzy_match=True, threshold=80):
        """
        Search for a medicine by name
        Returns complete information about the medicine
        
        Parameters:
        - query: Medicine name to search for
        - fuzzy_match: Whether to use fuzzy matching
        - threshold: Minimum matching score (0-100) for fuzzy matching
        
        Returns:
        - Dictionary containing medicine information or None if not found
        """
        query = query.lower().strip()
        
        # Direct match
        if query in self.medicine_dict:
            return self.medicine_dict[query]
        
        # Fuzzy match if enabled
        if fuzzy_match and self.medicine_dict:
            matches = process.extractOne(query, self.medicine_dict.keys(), 
                                        scorer=fuzz.token_sort_ratio)
            if matches and matches[1] >= threshold:
                best_match = matches[0]
                print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.medicine_dict[best_match]
        
        print(f"Medicine '{query}' not found")
        return None
    
    def search_drug(self, query, fuzzy_match=True, threshold=80):
        """
        Search for a drug by name
        Returns drug properties and side effects
        
        Parameters:
        - query: Drug name to search for
        - fuzzy_match: Whether to use fuzzy matching
        - threshold: Minimum matching score (0-100) for fuzzy matching
        
        Returns:
        - Dictionary containing drug information or None if not found
        """
        query = query.lower().strip()
        
        # Direct match
        if query in self.drug_dict:
            result = self.drug_dict[query].copy()
            
            # Add side effects from medicines containing this drug
            if query in self.drug_to_medicines:
                side_effects = []
                for med_name in self.drug_to_medicines[query]:
                    if 'side_effects' in self.medicine_dict[med_name] and not pd.isna(self.medicine_dict[med_name]['side_effects']):
                        effects = self.medicine_dict[med_name]['side_effects']
                        side_effects.append(f"From {med_name}: {effects}")
                
                result['side_effects_from_medicines'] = side_effects
            
            # Add repurposing information if available
            if self.rep_data is not None:
                repurposed_uses = self.rep_data[self.rep_data['drug'].str.lower() == query] if 'drug' in self.rep_data.columns else None
                if repurposed_uses is not None and not repurposed_uses.empty:
                    result['repurposed_uses'] = repurposed_uses[['repurposed_use', 'confidence_level']].to_dict('records')
            
            return result
        
        # Fuzzy match if enabled
        if fuzzy_match and self.drug_dict:
            matches = process.extractOne(query, self.drug_dict.keys(),
                                        scorer=fuzz.token_sort_ratio)
            if matches and matches[1] >= threshold:
                best_match = matches[0]
                print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.search_drug(best_match, fuzzy_match=False)
        
        print(f"Drug '{query}' not found")
        return None
    
    def get_drugs_in_medicine(self, medicine_name):
        """
        Get all drugs that compose a specific medicine
        """
        medicine_name = medicine_name.lower().strip()
        
        # Check direct match
        if medicine_name in self.medicine_to_drugs:
            return self.medicine_to_drugs[medicine_name]
        
        # Try fuzzy matching
        matches = process.extractOne(medicine_name, self.medicine_to_drugs.keys(),
                                    scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.medicine_to_drugs[best_match]
        
        print(f"Medicine '{medicine_name}' not found")
        return []
    
    def get_medicines_containing_drug(self, drug_name):
        """
        Get all medicines containing a specific drug
        """
        drug_name = drug_name.lower().strip()
        
        # Check direct match
        if drug_name in self.drug_to_medicines:
            return self.drug_to_medicines[drug_name]
        
        # Try fuzzy matching
        matches = process.extractOne(drug_name, self.drug_to_medicines.keys(),
                                    scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.drug_to_medicines[best_match]
        
        print(f"Drug '{drug_name}' not found in any medicine")
        return []
    
    def search_by_side_effect(self, side_effect_query):
        """
        Find medicines and drugs associated with a specific side effect
        """
        if self.med_data is None or 'side_effects' not in self.med_data.columns:
            print("Side effects data not available")
            return []
        
        side_effect_query = side_effect_query.lower()
        results = []
        
        for idx, row in self.med_data.iterrows():
            if 'side_effects' in row and not pd.isna(row['side_effects']):
                if side_effect_query in str(row['side_effects']).lower():
                    med_info = {
                        'medicine_name': row['medicine_name'] if 'medicine_name' in row else 'Unknown',
                        'composition': row['composition'] if 'composition' in row else 'Unknown',
                        'side_effects': row['side_effects']
                    }
                    results.append(med_info)
        
        return results
    
    def get_stats(self):
        """
        Return statistics about the loaded data
        """
        stats = {
            'num_medicines': len(self.medicine_dict) if self.medicine_dict else 0,
            'num_unique_drugs': len(self.drug_dict) if self.drug_dict else 0,
            'num_repurposed_drugs': len(self.rep_data) if self.rep_data is not None else 0,
            'avg_drugs_per_medicine': sum(len(drugs) for drugs in self.medicine_to_drugs.values()) / len(self.medicine_to_drugs) if self.medicine_to_drugs else 0
        }
        return stats

# Example usage
def main():
    # Initialize search engine
    search_engine = DrugSearchEngine()
    
    # Print statistics
    stats = search_engine.get_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    
    # Example searches
    print("\n--- Example Medicine Search ---")
    medicine_query = "Aspirin"  # Replace with actual medicine from your dataset
    medicine_info = search_engine.search_medicine(medicine_query)
    if medicine_info:
        print(f"Found medicine: {medicine_query}")
        print(f"Composition: {medicine_info.get('composition', 'N/A')}")
        print(f"Uses: {medicine_info.get('uses', 'N/A')}")
        print(f"Side effects: {medicine_info.get('side_effects', 'N/A')}")
    
    print("\n--- Example Drug Search ---")
    drug_query = "Paracetamol"  # Replace with actual drug from your dataset
    drug_info = search_engine.search_drug(drug_query)
    if drug_info:
        print(f"Found drug: {drug_query}")
        print(f"SMILES: {drug_info.get('smiles', 'N/A')}")
        print(f"Molecular weight: {drug_info.get('molecular_weight', 'N/A')}")
        print(f"Side effects from medicines:")
        for effect in drug_info.get('side_effects_from_medicines', ['None found']):
            print(f"  - {effect}")
    
    # Find medicines containing a drug
    print("\n--- Medicines Containing Drug ---")
    containing_drug = search_engine.get_medicines_containing_drug(drug_query)
    if containing_drug:
        print(f"Medicines containing {drug_query}:")
        for med in containing_drug:
            print(f"- {med}")
    
    # Search by side effect
    print("\n--- Search by Side Effect ---")
    side_effect_query = "headache"  # Replace with relevant side effect
    side_effect_results = search_engine.search_by_side_effect(side_effect_query)
    if side_effect_results:
        print(f"Medicines with side effect '{side_effect_query}':")
        for result in side_effect_results:
            print(f"- {result['medicine_name']} ({result['composition']})")

if __name__ == "__main__":
    main()