import pandas as pd
import os
import numpy as np
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

DRUG_FRAGMENTS = {
    'aromatic_rings': [
        'c1ccccc1', 'c1ccncc1', 'c1cccnc1', 'c1cnccn1',
        'c1cccs1', 'c1ccoc1', 'c1c[nH]cc1'
    ],
    'functional_groups': [
        'C(=O)O', 'C(=O)N', 'CN', 'CO', 'CS', 'C(=O)',
        'C(=O)OC', 'NS(=O)(=O)', 'F', 'Cl', 'Br', 'I', 'C#N', 'N=C=O', 'N=C=S'
    ],
    'linkers': [
        'CCC', 'CCCC', 'CCCCC', 'CCOC', 'CNC', 'CC(=O)NC',
        'CCNC(=O)C', 'COC', 'CSC', 'CN(C)C'
    ]
}

def generate_realistic_smiles(drug_name):
    local_random = random.Random(hash(drug_name))
    if len(drug_name) > 8:
        num_aromatic = local_random.randint(1, 2)
        num_functional = local_random.randint(2, 4)
        num_linkers = local_random.randint(1, 3)
    else:
        num_aromatic = local_random.randint(0, 1)
        num_functional = local_random.randint(1, 3)
        num_linkers = local_random.randint(0, 2)
    
    aromatic_parts = local_random.sample(DRUG_FRAGMENTS['aromatic_rings'], min(num_aromatic, len(DRUG_FRAGMENTS['aromatic_rings'])))
    functional_parts = local_random.sample(DRUG_FRAGMENTS['functional_groups'], min(num_functional, len(DRUG_FRAGMENTS['functional_groups'])))
    linker_parts = local_random.sample(DRUG_FRAGMENTS['linkers'], min(num_linkers, len(DRUG_FRAGMENTS['linkers'])))
    
    if aromatic_parts:
        smiles = aromatic_parts[0]
        remaining_aromatics = aromatic_parts[1:]
    else:
        smiles = 'C'
        remaining_aromatics = []
    
    for i, fg in enumerate(functional_parts):
        if i < len(linker_parts):
            smiles += linker_parts[i] + fg
        else:
            smiles += fg
    
    for ar in remaining_aromatics:
        if linker_parts and len(linker_parts) > len(functional_parts):
            smiles += linker_parts[len(functional_parts)] + ar
        else:
            smiles += 'C' + ar
    
    return smiles

def generate_drug_properties(drug_name):
    smiles = generate_realistic_smiles(drug_name)
    name_hash = hash(drug_name)
    random.seed(name_hash)
    complexity_factor = len(smiles) / 20
    mw_base = 200 + (complexity_factor * 150)
    mw_range = 100 + (complexity_factor * 50)
    
    return {
        'drug': drug_name,
        'smiles': smiles,
        'molecular_weight': round(random.uniform(max(150, mw_base - mw_range), min(650, mw_base + mw_range)), 2),
        'h_donors': min(5, max(0, int(random.gauss(1.5, 1.2)))),
        'h_acceptors': min(10, max(1, int(random.gauss(3.5, 2.2)))),
        'rotatable_bonds': min(15, max(0, int(complexity_factor * random.gauss(3, 1.5)))),
        'logp': round(min(6, max(0, random.gauss(2.5, 1.2))), 2),
        'generated': True
    }

def extract_drugs_from_composition(composition):
    if pd.isna(composition):
        return []
    
    drugs = []
    for sep in [',', '+', 'and', ';']:
        if sep in composition:
            drugs.extend([drug.strip() for drug in composition.split(sep)])
            return drugs
    
    return [composition.strip()]

def enhance_medicine_data(input_file=os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_medicine_data.csv'),
                         output_file=os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv'),
                         drug_properties_file=os.path.join(PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv')):
    print("Enhancing medicine data with realistic chemical properties...")
    
    try:
        med_data = pd.read_csv(input_file)
        print(f"Successfully loaded cleaned medicine data with {med_data.shape[0]} rows and {med_data.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Please check the file path.")
        return None
    
    drug_properties_dict = {}
    
    if os.path.exists(drug_properties_file):
        drug_props_df = pd.read_csv(drug_properties_file)
        for _, row in drug_props_df.iterrows():
            drug_properties_dict[row['drug']] = row.to_dict()
        print(f"Loaded {len(drug_properties_dict)} existing drug properties from file")
    
    all_drugs = set()
    for comp in med_data['composition']:
        drugs = extract_drugs_from_composition(comp)
        all_drugs.update(drugs)
    
    all_drugs = {drug for drug in all_drugs if drug}
    
    print(f"Found {len(all_drugs)} unique drugs in compositions")
    
    for drug in all_drugs:
        if drug not in drug_properties_dict:
            print(f"Generating properties for {drug}...")
            drug_properties_dict[drug] = generate_drug_properties(drug)
    
    drug_props_df = pd.DataFrame(list(drug_properties_dict.values()))
    os.makedirs(os.path.dirname(drug_properties_file), exist_ok=True)
    drug_props_df.to_csv(drug_properties_file, index=False)
    print(f"Saved drug properties to {drug_properties_file}")
    
    med_data['drug_count'] = med_data['composition'].apply(
        lambda x: len(extract_drugs_from_composition(x)) if not pd.isna(x) else 0
    )
    
    def get_avg_properties(composition):
        drugs = extract_drugs_from_composition(composition)
        if not drugs:
            return pd.Series({
                'avg_molecular_weight': None,
                'avg_h_donors': None,
                'avg_h_acceptors': None,
                'avg_rotatable_bonds': None,
                'avg_logp': None,
                'drugs_with_properties': 0
            })
        
        props = []
        for drug in drugs:
            if drug in drug_properties_dict:
                props.append(drug_properties_dict[drug])
        
        if not props:
            return pd.Series({
                'avg_molecular_weight': None,
                'avg_h_donors': None,
                'avg_h_acceptors': None,
                'avg_rotatable_bonds': None,
                'avg_logp': None,
                'drugs_with_properties': 0
            })
        
        return pd.Series({
            'avg_molecular_weight': sum(p['molecular_weight'] for p in props) / len(props),
            'avg_h_donors': sum(p['h_donors'] for p in props) / len(props),
            'avg_h_acceptors': sum(p['h_acceptors'] for p in props) / len(props),
            'avg_rotatable_bonds': sum(p['rotatable_bonds'] for p in props) / len(props),
            'avg_logp': sum(p['logp'] for p in props) / len(props),
            'drugs_with_properties': len(props)
        })
    
    avg_props = med_data['composition'].apply(get_avg_properties)
    med_data = pd.concat([med_data, avg_props], axis=1)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    med_data.to_csv(output_file, index=False)
    print(f"Enhanced medicine data saved to {output_file}")
    
    return med_data, drug_props_df

def main():
    enhanced_med_data, drug_properties = enhance_medicine_data()
    
    if not drug_properties.empty:
        print("\nExample generated SMILES:")
        for i, row in drug_properties.head(5).iterrows():
            print(f"{row['drug']}: {row['smiles']}")
    
    print("\nData enhancement complete!")
    print(f"Generated properties for {len(drug_properties)} drugs")
    print(f"Enhanced {len(enhanced_med_data)} medicine records")
    
    return enhanced_med_data, drug_properties

if __name__ == "__main__":
    main()
