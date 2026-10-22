import pandas as pd
import os
from fuzzywuzzy import process, fuzz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DrugSearchEngine:
    def __init__(self, 
                 medicine_data_path=None,
                 drug_properties_path=None,
                 repurposing_data_path=None):
        medicine_data_path = medicine_data_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv')
        drug_properties_path = drug_properties_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv')
        repurposing_data_path = repurposing_data_path or os.path.join(
            PROJECT_ROOT, 'data', 'processed', 'cleaned_repurposing_data.csv')
        self.med_data = pd.read_csv(medicine_data_path) if os.path.exists(medicine_data_path) else None
        self.drug_props = pd.read_csv(drug_properties_path) if os.path.exists(drug_properties_path) else None
        self.rep_data = pd.read_csv(repurposing_data_path) if os.path.exists(repurposing_data_path) else None
        
        self.medicine_dict = {}
        self.drug_dict = {}
        self.medicine_to_drugs = {}
        self.drug_to_medicines = {}
        
        if self.med_data is not None:
            for idx, row in self.med_data.iterrows():
                med_name = row['medicine_name'].lower() if 'medicine_name' in row else None
                if med_name:
                    self.medicine_dict[med_name] = row.to_dict()
                    if 'composition' in row and not pd.isna(row['composition']):
                        drugs = self._extract_drugs_from_composition(row['composition'])
                        self.medicine_to_drugs[med_name] = drugs
                        for drug in drugs:
                            if drug not in self.drug_to_medicines:
                                self.drug_to_medicines[drug] = []
                            self.drug_to_medicines[drug].append(med_name)
        
        if self.drug_props is not None:
            for idx, row in self.drug_props.iterrows():
                drug_name = row['drug'].lower() if 'drug' in row else None
                if drug_name:
                    self.drug_dict[drug_name] = row.to_dict()
    
    def _extract_drugs_from_composition(self, composition):
        if pd.isna(composition):
            return []
        drugs = []
        for sep in [',', '+', 'and', ';']:
            if sep in composition:
                drugs.extend([drug.strip().lower() for drug in composition.split(sep)])
                return drugs
        return [composition.strip().lower()]
    
    def search_medicine(self, query, fuzzy_match=True, threshold=80):
        query = query.lower().strip()
        if query in self.medicine_dict:
            return self.medicine_dict[query]
        if fuzzy_match and self.medicine_dict:
            matches = process.extractOne(query, self.medicine_dict.keys(), scorer=fuzz.token_sort_ratio)
            if matches and matches[1] >= threshold:
                best_match = matches[0]
                print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.medicine_dict[best_match]
        print(f"Medicine '{query}' not found")
        return None
    
    def search_drug(self, query, fuzzy_match=True, threshold=80):
        query = query.lower().strip()
        if query in self.drug_dict:
            return self.drug_dict[query]
        if fuzzy_match and self.drug_dict:
            matches = process.extractOne(query, self.drug_dict.keys(), scorer=fuzz.token_sort_ratio)
            if matches and matches[1] >= threshold:
                best_match = matches[0]
                print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.search_drug(best_match, fuzzy_match=False)
        print(f"Drug '{query}' not found")
        return None
    
    def get_drugs_in_medicine(self, medicine_name):
        medicine_name = medicine_name.lower().strip()
        if medicine_name in self.medicine_to_drugs:
            return self.medicine_to_drugs[medicine_name]
        matches = process.extractOne(medicine_name, self.medicine_to_drugs.keys(), scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.medicine_to_drugs[best_match]
        print(f"Medicine '{medicine_name}' not found")
        return []
    
    def get_medicines_containing_drug(self, drug_name):
        drug_name = drug_name.lower().strip()
        if drug_name in self.drug_to_medicines:
            return self.drug_to_medicines[drug_name]
        matches = process.extractOne(drug_name, self.drug_to_medicines.keys(), scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            print(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.drug_to_medicines[best_match]
        print(f"Drug '{drug_name}' not found in any medicine")
        return []
    
    def search_by_side_effect(self, side_effect_query):
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
                        'composition': row['composition'] if 'composition' in row else 'Unknown'
                    }
                    results.append(med_info)
        return results
    
    def get_stats(self):
        stats = {
            'num_medicines': len(self.medicine_dict) if self.medicine_dict else 0,
            'num_unique_drugs': len(self.drug_dict) if self.drug_dict else 0,
            'num_repurposed_drugs': len(self.rep_data) if self.rep_data is not None else 0,
            'avg_drugs_per_medicine': sum(len(drugs) for drugs in self.medicine_to_drugs.values()) / len(self.medicine_to_drugs) if self.medicine_to_drugs else 0
        }
        return stats

def display_drug_info(drug_info):
    if not drug_info:
        return
    print("\n===== DRUG INFORMATION =====")
    print(f"Name: {drug_info.get('drug', 'N/A')}")
    print(f"SMILES: {drug_info.get('smiles', 'N/A')}")
    print(f"Molecular Weight: {drug_info.get('molecular_weight', 'N/A')}")
    print(f"LogP: {drug_info.get('logp', 'N/A')}")
    print(f"H-Bond Donors: {drug_info.get('h_donors', 'N/A')}")
    print(f"H-Bond Acceptors: {drug_info.get('h_acceptors', 'N/A')}")
    print("=============================")

def display_medicine_info(medicine_info):
    if not medicine_info:
        return
    print("\n===== MEDICINE INFORMATION =====")
    print(f"Name: {medicine_info.get('medicine_name', 'N/A')}")
    print(f"Composition: {medicine_info.get('composition', 'N/A')}")
    print(f"good review: {medicine_info.get('excellent_review_%', 'N/A')}%")
    print(f"bad review: {medicine_info.get('poor_review_%', 'N/A')}%")
    print(f"side effects: {medicine_info.get('side_effects', 'N/A')}")
    if 'uses' in medicine_info and not pd.isna(medicine_info['uses']):
        print(f"\nUses: {medicine_info['uses']}")
    else:
        print("\nNo uses data available")
    print("===============================")

def interactive_menu():
    search_engine = DrugSearchEngine()
    print("\n" + "="*50)
    print("WELCOME TO THE DRUG SEARCH SYSTEM")
    print("="*50)
    stats = search_engine.get_stats()
    print("\nDatabase Statistics:")
    print(f"- Total Medicines: {stats['num_medicines']}")
    print(f"- Total Unique Drugs: {stats['num_unique_drugs']}")
    print(f"- Repurposed Drugs: {stats['num_repurposed_drugs']}")
    print(f"- Average Drugs per Medicine: {stats['avg_drugs_per_medicine']:.2f}")
    
    while True:
        print("\n" + "-"*50)
        print("MAIN MENU")
        print("-"*50)
        print("1. Search for a Drug")
        print("2. Search for a Medicine")
        print("3. Find Medicines Containing a Drug")
        print("4. Find Drugs in a Medicine")
        print("5. Search by Side Effect")
        print("6. Show Database Statistics")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ")
        
        if choice == '0':
            print("\nThank you for using the Drug Search System. Goodbye!")
            break
        elif choice == '1':
            drug_name = input("\nEnter drug name to search: ")
            drug_info = search_engine.search_drug(drug_name)
            display_drug_info(drug_info)
        elif choice == '2':
            medicine_name = input("\nEnter medicine name to search: ")
            medicine_info = search_engine.search_medicine(medicine_name)
            display_medicine_info(medicine_info)
        elif choice == '3':
            drug_name = input("\nEnter drug name to find medicines containing it: ")
            medicines = search_engine.get_medicines_containing_drug(drug_name)
            if medicines:
                print(f"\nMedicines containing {drug_name}:")
                for i, med in enumerate(medicines, 1):
                    print(f"{i}. {med}")
            else:
                print(f"No medicines found containing {drug_name}.")
        elif choice == '4':
            medicine_name = input("\nEnter medicine name to find its drugs: ")
            drugs = search_engine.get_drugs_in_medicine(medicine_name)
            if drugs:
                print(f"\nDrugs in {medicine_name}:")
                for i, drug in enumerate(drugs, 1):
                    print(f"{i}. {drug}")
            else:
                print(f"No drugs found in {medicine_name}.")
        elif choice == '5':
            side_effect = input("\nEnter side effect to search for: ")
            results = search_engine.search_by_side_effect(side_effect)
            if results:
                print(f"\nMedicines with side effect '{side_effect}':")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['medicine_name']} - {result['composition']}")
            else:
                print(f"No medicines found with side effect '{side_effect}'")
        elif choice == '6':
            stats = search_engine.get_stats()
            print("\nDatabase Statistics:")
            print(f"- Total Medicines: {stats['num_medicines']}")
            print(f"- Total Unique Drugs: {stats['num_unique_drugs']}")
            print(f"- Repurposed Drugs: {stats['num_repurposed_drugs']}")
            print(f"- Average Drugs per Medicine: {stats['avg_drugs_per_medicine']:.2f}")
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        interactive_menu()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure that all the required data files exist at the specified paths.")
