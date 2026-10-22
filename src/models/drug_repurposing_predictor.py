import pandas as pd
import numpy as np
import pickle
import re
import time
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, precision_score, recall_score
from tqdm import tqdm

# Adding Neural Network imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress TensorFlow/oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DrugRepurposingPredictor:
    def __init__(self):
        """
        Initialize the DrugRepurposingPredictor with Neural Network model
        """
        self.text_vectorizer = TfidfVectorizer(max_features=5000)
        self.model_type = 'nn'  # Always use neural network
        self.classifier = None  # Will be built during training
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()  # For numerical molecular features
        self.training_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'hamming_loss': [],
            'iterations': []
        }
        
    def preprocess_conditions(self, conditions_text):
        """Extract individual conditions from the conditions text"""
        if pd.isna(conditions_text):
            return []
            
        # Split by major delimiters
        conditions = re.split(r', |; |\. |Treatment of |Prevention of ', conditions_text)
        
        # Clean up each condition
        conditions = [re.sub(r'\s+', ' ', condition).strip() for condition in conditions if condition.strip()]
        
        # Remove any duplicates while preserving order
        seen = set()
        conditions = [x for x in conditions if not (x in seen or seen.add(x))]
        
        return conditions
    
    def load_and_merge_data(self, drug_props_path, medicine_data_path):
        """
        Load and merge data from the two datasets
        Returns a dataframe ready for model training
        """
        print("Loading data from real datasets...")
        
        # Load the datasets
        drug_props = pd.read_csv(drug_props_path)
        medicine_data = pd.read_csv(medicine_data_path)
        
        # Basic data cleaning
        # Remove parentheses and dosage info from drug names in drug_props
        drug_props['drug'] = drug_props['drug'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip())
        
        # Create a dictionary of drugs with their properties for easy lookup
        drug_dict = {}
        for _, row in drug_props.iterrows():
            drug_dict[row['drug'].lower()] = {
                'smiles': row['smiles'],
                'molecular_weight': row['molecular_weight'],
                'h_donors': row['h_donors'],
                'h_acceptors': row['h_acceptors'],
                'rotatable_bonds': row['rotatable_bonds'],
                'logp': row['logp']
            }
        
        # Create a dataframe for modeling
        model_data = []
        
        # Process each medicine and extract drug components
        for _, medicine in medicine_data.iterrows():
            # Extract individual drugs from composition
            if pd.isna(medicine['composition']):
                continue
                
            composition = medicine['composition']
            drug_components = re.split(r'\+|\,', composition)
            
            # Process each drug component
            for drug_component in drug_components:
                # Extract drug name (removing dosage)
                drug_name = re.sub(r'\s*\([^)]*\)', '', drug_component).strip()
                
                # Look for the drug in our properties dictionary
                drug_key = None
                for key in drug_dict:
                    if key.lower() in drug_name.lower() or drug_name.lower() in key.lower():
                        drug_key = key
                        break
                
                if drug_key:
                    # We found a match, create a data entry
                    entry = {
                        'drug_name': drug_name,
                        'chemical_structure': drug_dict[drug_key]['smiles'],
                        'molecular_weight': drug_dict[drug_key]['molecular_weight'],
                        'h_donors': drug_dict[drug_key]['h_donors'],
                        'h_acceptors': drug_dict[drug_key]['h_acceptors'],
                        'rotatable_bonds': drug_dict[drug_key]['rotatable_bonds'],
                        'logp': drug_dict[drug_key]['logp'],
                        'known_conditions': medicine['uses']
                    }
                    model_data.append(entry)
        
        # If we don't have enough matches, add all drugs with dummy conditions
        if len(model_data) < 10:
            print("Not enough matching drugs found. Adding all drugs with placeholder conditions...")
            for _, drug in drug_props.iterrows():
                entry = {
                    'drug_name': drug['drug'],
                    'chemical_structure': drug['smiles'],
                    'molecular_weight': drug['molecular_weight'],
                    'h_donors': drug['h_donors'],
                    'h_acceptors': drug['h_acceptors'],
                    'rotatable_bonds': drug['rotatable_bonds'],
                    'logp': drug['logp'],
                    'known_conditions': "Various medical conditions"  # Placeholder
                }
                model_data.append(entry)
                
        # Create DataFrame
        df = pd.DataFrame(model_data)
        
        # Add molecular_properties field for backward compatibility
        df['molecular_properties'] = df.apply(
            lambda row: f"MW: {row['molecular_weight']}, HBD: {row['h_donors']}, HBA: {row['h_acceptors']}, " \
                        f"RotB: {row['rotatable_bonds']}, LogP: {row['logp']}",
            axis=1
        )
        
        print(f"Created dataset with {len(df)} drug entries")
        return df
    
    def extract_molecular_features(self, df):
        """Extract structured numerical features from molecular properties"""
        # This function is still needed for loading saved models, but for new data
        # we'll already have the features extracted
        
        # Check if numerical columns already exist
        if all(col in df.columns for col in ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']):
            print("Numerical features already present in dataset")
            return df
            
        # Parse the molecular properties into numerical features
        # Initialize new numerical columns
        df['molecular_weight'] = np.nan
        df['h_donors'] = np.nan
        df['h_acceptors'] = np.nan
        df['rotatable_bonds'] = np.nan
        df['logp'] = np.nan
        
        # Extract numerical features from molecular_properties text
        for idx, props in enumerate(df['molecular_properties']):
            try:
                # Extract molecular weight
                mw_match = re.search(r'MW[:\s]*(\d+\.?\d*)', props)
                if mw_match:
                    df.at[idx, 'molecular_weight'] = float(mw_match.group(1))
                
                # Extract H-donors
                hd_match = re.search(r'HBD[:\s]*(\d+)', props)
                if hd_match:
                    df.at[idx, 'h_donors'] = int(hd_match.group(1))
                
                # Extract H-acceptors
                ha_match = re.search(r'HBA[:\s]*(\d+)', props)
                if ha_match:
                    df.at[idx, 'h_acceptors'] = int(ha_match.group(1))
                
                # Extract rotatable bonds
                rb_match = re.search(r'RotB[:\s]*(\d+)', props)
                if rb_match:
                    df.at[idx, 'rotatable_bonds'] = int(rb_match.group(1))
                
                # Extract LogP
                logp_match = re.search(r'LogP[:\s]*(-?\d+\.?\d*)', props)
                if logp_match:
                    df.at[idx, 'logp'] = float(logp_match.group(1))
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        # Fill missing values with median
        for col in ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']:
            if df[col].isna().all():
                df[col] = 0  # If all values are missing, use 0
            else:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _build_improved_neural_network(self, text_input_dim, numerical_input_dim, output_dim):
        """Build an improved neural network model for multi-label classification with dual inputs"""
        # Clear any existing Keras session
        tf.keras.backend.clear_session()
        
        # Text input branch
        text_input = Input(shape=(text_input_dim,), name='text_input')
        text_branch = Dense(512, activation='relu')(text_input)
        text_branch = Dropout(0.3)(text_branch)
        text_branch = Dense(256, activation='relu')(text_branch)
        text_branch = Dropout(0.3)(text_branch)
        
        # Numerical input branch
        numerical_input = Input(shape=(numerical_input_dim,), name='numerical_input')
        numerical_branch = Dense(64, activation='relu')(numerical_input)
        numerical_branch = Dropout(0.2)(numerical_branch)
        
        # Merge branches
        merged = Concatenate()([text_branch, numerical_branch])
        
        # Common network after merging
        x = Dense(512, activation='relu')(merged)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        output = Dense(output_dim, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=[text_input, numerical_input], outputs=output)
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
        
    def train(self, drug_props_path, medicine_data_path, verbose=False, track_progress=True):
        """Train the model on the drug data with progress tracking"""
        # Load and merge data
        df = self.load_and_merge_data(drug_props_path, medicine_data_path)
        
        # Create text features from drug name and chemical structure
        df['text_features'] = df['drug_name'] + ' ' + df['chemical_structure']
        
        # Extract and prepare the labels (known conditions)
        print("Preparing features and labels...")
        conditions_list = df['known_conditions'].apply(self.preprocess_conditions)
        
        # Transform labels to binary format
        y = self.mlb.fit_transform(conditions_list)
        
        # Transform text features
        X_text = self.text_vectorizer.fit_transform(df['text_features'])
        
        # Prepare numerical features
        numerical_cols = ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']
        X_numerical = df[numerical_cols].values
        
        # Scale numerical features
        X_numerical = self.scaler.fit_transform(X_numerical)
        
        # Split data
        # First split text features
        X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42
        )
        
        # Then split numerical features using the same indices
        # Get the indices from the train_test_split
        all_indices = np.arange(len(df))
        _, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
        train_indices = np.array([i for i in all_indices if i not in test_indices])
        
        X_numerical_train = X_numerical[train_indices]
        X_numerical_test = X_numerical[test_indices]

        # Reset training history
        self.training_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'hamming_loss': [],
            'iterations': []
        }
        
        # Convert sparse matrix to dense for Keras
        X_text_train_dense = X_text_train.toarray()
        X_text_test_dense = X_text_test.toarray()
        
        # Build improved neural network model
        text_input_dim = X_text_train_dense.shape[1]
        numerical_input_dim = X_numerical_train.shape[1]
        output_dim = y_train.shape[1]
        
        print(f"Building improved Neural Network model with:")
        print(f"- Text input dim: {text_input_dim}")
        print(f"- Numerical input dim: {numerical_input_dim}")
        print(f"- Output dim: {output_dim}")
        
        self.classifier = self._build_improved_neural_network(
            text_input_dim, 
            numerical_input_dim, 
            output_dim
        )
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        )
        
        # Train model with progress bar
        epochs = 30
        batch_size = 32
        
        if track_progress:
            # Create custom callback to track metrics during training
            class MetricsCallback(tf.keras.callbacks.Callback):
                def __init__(self, X_text_test, X_numerical_test, y_test, training_history, verbose=False):
                    super().__init__()
                    self.X_text_test = X_text_test
                    self.X_numerical_test = X_numerical_test
                    self.y_test = y_test
                    self.training_history = training_history
                    self.verbose = verbose
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Make predictions
                    y_pred = (self.model.predict(
                        [self.X_text_test, self.X_numerical_test], 
                        verbose=0
                    ) > 0.5).astype(int)
                    
                    # Calculate metrics
                    h_loss = hamming_loss(self.y_test, y_pred)
                    acc = accuracy_score(self.y_test, y_pred)
                    precision = precision_score(self.y_test, y_pred, average='samples', zero_division=0)
                    recall = recall_score(self.y_test, y_pred, average='samples', zero_division=0)
                    
                    # Update training history
                    self.training_history['accuracy'].append(acc)
                    self.training_history['precision'].append(precision)
                    self.training_history['recall'].append(recall)
                    self.training_history['hamming_loss'].append(h_loss)
                    self.training_history['iterations'].append(epoch + 1)
                    
                    if self.verbose:
                        print(f"Epoch {epoch+1}: Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                              f"Recall: {recall:.4f}, Hamming Loss: {h_loss:.6f}")
            
            # Initialize and use the custom callback
            metrics_callback = MetricsCallback(
                X_text_test_dense, 
                X_numerical_test, 
                y_test, 
                self.training_history,
                verbose=verbose
            )
            
            # Train with history recording
            history = self.classifier.fit(
                [X_text_train_dense, X_numerical_train], y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_text_test_dense, X_numerical_test], y_test),
                callbacks=[early_stopping, metrics_callback],
                verbose=1 if verbose else 0
            )
        else:
            # Train without detailed history
            self.classifier.fit(
                [X_text_train_dense, X_numerical_train], y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_text_test_dense, X_numerical_test], y_test),
                callbacks=[early_stopping],
                verbose=0
            )
        
        print("Training complete!")
        
        # Calculate final metrics
        y_pred = (self.classifier.predict(
            [X_text_test_dense, X_numerical_test], 
            verbose=0
        ) > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='samples', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='samples', zero_division=0),
            'hamming_loss': hamming_loss(y_test, y_pred),
            'classes': self.mlb.classes_
        }
        
        # Add final point to history if needed
        if track_progress and len(self.training_history['iterations']) < epochs:
            self.training_history['accuracy'].append(metrics['accuracy'])
            self.training_history['precision'].append(metrics['precision'])
            self.training_history['recall'].append(metrics['recall'])
            self.training_history['hamming_loss'].append(metrics['hamming_loss'])
            self.training_history['iterations'].append(len(self.training_history['iterations']) + 1)
        
        if verbose:
            print("\nModel Evaluation:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"Number of unique conditions: {len(self.mlb.classes_)}")
        
        return metrics
    
    def predict_repurposing(self, drug_name, chemical_structure, molecular_features, confidence_threshold=0.2, top_n=5):
        """Predict new therapeutic uses for an existing drug"""
        # Combine text features
        text_features = drug_name + ' ' + chemical_structure
        
        # Transform text input
        X_text = self.text_vectorizer.transform([text_features])
        X_text_dense = X_text.toarray()
        
        # Process numerical features
        X_numerical = np.array([[
            molecular_features['molecular_weight'],
            molecular_features['h_donors'],
            molecular_features['h_acceptors'],
            molecular_features['rotatable_bonds'],
            molecular_features['logp']
        ]])
        X_numerical = self.scaler.transform(X_numerical)
        
        # Neural Network prediction - get raw probabilities
        pred_proba = self.classifier.predict([X_text_dense, X_numerical], verbose=0)[0]
        
        # Get indices sorted by probability (highest first)
        sorted_indices = np.argsort(pred_proba)[::-1]
        
        # Create a list of (condition, probability) tuples
        predictions = []
        for idx in sorted_indices:
            if pred_proba[idx] >= confidence_threshold:
                condition = self.mlb.classes_[idx]
                probability = pred_proba[idx]
                predictions.append((condition, probability))
        
        # Show some predictions even if confidence is low
        if len(predictions) < top_n:
            remaining = top_n - len(predictions)
            for idx in sorted_indices:
                condition = self.mlb.classes_[idx]
                probability = pred_proba[idx]
                if (condition, probability) not in predictions:
                    predictions.append((condition, probability))
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        # Take only top_n predictions
        predictions = predictions[:top_n]
        
        return predictions
    
    def save_model(self, path):
        """Save the trained model to disk"""
        # For neural network, save the model separately with proper extension
        model_path = path.replace('.pkl', '_nn_model.keras')
        self.classifier.save(model_path)
        
        # Save everything except the classifier
        model_data = {
            'text_vectorizer': self.text_vectorizer,
            'mlb': self.mlb,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'model_type': self.model_type,
            'nn_model_path': model_path
        }
            
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path):
        """Load a trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = DrugRepurposingPredictor()
        
        # Load components
        model.text_vectorizer = model_data['text_vectorizer']
        model.mlb = model_data['mlb']
        
        if 'scaler' in model_data:
            model.scaler = model_data['scaler']
        
        if 'training_history' in model_data:
            model.training_history = model_data['training_history']
            
        # Load neural network model
        nn_path = model_data.get('nn_model_path')
        if nn_path:
            try:
                model.classifier = tf.keras.models.load_model(nn_path)
            except FileNotFoundError:
                # If the path is different, try relative path
                base_dir = os.path.dirname(path)
                model_name = os.path.basename(nn_path)
                new_path = os.path.join(base_dir, model_name)
                model.classifier = tf.keras.models.load_model(new_path)
        else:
            raise ValueError("Neural network model path not found in saved data")
        
        return model
    
    def evaluate(self, drug_props_path, medicine_data_path):
        """Evaluate the model on the given dataset with progress visualization"""
        # Load data
        df = self.load_and_merge_data(drug_props_path, medicine_data_path)
        
        # Create text features from drug name and chemical structure
        df['text_features'] = df['drug_name'] + ' ' + df['chemical_structure']
        
        # Extract and prepare the labels (known conditions)
        conditions_list = df['known_conditions'].apply(self.preprocess_conditions)
        
        # Transform labels to binary format
        y_true = self.mlb.transform(conditions_list)
        
        # Transform text features
        X_text = self.text_vectorizer.transform(df['text_features'])
        X_text_dense = X_text.toarray()
        
        # Prepare numerical features
        numerical_cols = ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']
        X_numerical = df[numerical_cols].values
        X_numerical = self.scaler.transform(X_numerical)
        
        # Make predictions
        print("Evaluating model...")
        with tqdm(total=100, desc="Evaluation Progress") as pbar:
            pbar.update(50)
            
            # Neural network prediction
            y_pred = (self.classifier.predict(
                [X_text_dense, X_numerical], 
                verbose=0
            ) > 0.5).astype(int)
                
            pbar.update(50)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
        h_loss = hamming_loss(y_true, y_pred)
        
        # Create and display graphs showing the model progress and final metrics
        self._plot_evaluation_with_progress(accuracy, precision, recall, h_loss)
        
        # Additional analysis - summary only
        print("\nOverall metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Hamming Loss: {h_loss:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'hamming_loss': h_loss
        }
    
    def _plot_evaluation_with_progress(self, final_accuracy, final_precision, final_recall, final_loss):
        """Plot model evaluation metrics and training progress"""
        try:
            # Create directory for results if it doesn't exist
            os.makedirs('C:/machine learning/drug_repurposing/results/figures', exist_ok=True)
            
            # Create figure with 2x2 layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot accuracy progress
            if self.training_history['iterations']:
                # Accuracy plot
                ax1.plot(self.training_history['iterations'], self.training_history['accuracy'], 
                         marker='o', linestyle='-', color='green', label='Training Progress')
                ax1.axhline(y=final_accuracy, color='blue', linestyle='--', label=f'Final: {final_accuracy:.4f}')
                ax1.set_title('Accuracy Progress During Training')
                ax1.set_xlabel('Training Epochs')
                ax1.set_ylabel('Accuracy')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
                
                # Precision plot
                ax2.plot(self.training_history['iterations'], self.training_history['precision'], 
                         marker='o', linestyle='-', color='purple', label='Training Progress')
                ax2.axhline(y=final_precision, color='blue', linestyle='--', label=f'Final: {final_precision:.4f}')
                ax2.set_title('Precision Progress During Training')
                ax2.set_xlabel('Training Epochs')
                ax2.set_ylabel('Precision')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
                
                # Recall plot
                ax3.plot(self.training_history['iterations'], self.training_history['recall'], 
                         marker='o', linestyle='-', color='orange', label='Training Progress')
                ax3.axhline(y=final_recall, color='blue', linestyle='--', label=f'Final: {final_recall:.4f}')
                ax3.set_title('Recall Progress During Training')
                ax3.set_xlabel('Training Epochs')
                ax3.set_ylabel('Recall')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()
                
                # Calculate appropriate y-axis limits for Hamming Loss
                min_loss = min(self.training_history['hamming_loss'])
                max_loss = max(self.training_history['hamming_loss'])
                
                # Add some padding to the limits
                loss_range = max_loss - min_loss
                if loss_range > 0:
                    padding = 0.1 * loss_range
                    y_min = max(0, min_loss - padding)  # Ensure it doesn't go below 0
                    y_max = max_loss + padding
                else:
                    # If range is very small, create an artificial range
                    base_value = min_loss if min_loss > 0 else 0.001
                    y_min = base_value * 0.8
                    y_max = base_value * 1.2
                
                # Hamming loss progress with adjusted y-axis
                ax4.plot(self.training_history['iterations'], self.training_history['hamming_loss'], 
                         marker='o', linestyle='-', color='red', label='Training Progress')
                ax4.axhline(y=final_loss, color='blue', linestyle='--', label=f'Final: {final_loss:.6f}')
                ax4.set_title('Hamming Loss Progress During Training')
                ax4.set_xlabel('Training Epochs')
                ax4.set_ylabel('Hamming Loss (Lower is Better)')
                ax4.set_ylim(y_min, y_max)  # Set the calculated limits
                ax4.grid(True, linestyle='--', alpha=0.7)
                ax4.legend()
                
            else:
                # No progress data available
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, "No training progress data available", 
                             ha='center', va='center')
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig('C:/machine learning/new/results/figures/model_evaluation_progress_nn1.png')
            print(f"Evaluation graph saved to C:/machine learning/new/results/figures/model_evaluation_progress_nn1.png")
            
            # Show the plot
            plt.show()
        except Exception as e:
            print(f"Could not create evaluation graph: {e}")
            print("Please install matplotlib with 'pip install matplotlib' to enable graphs.")


def check_model_exists(model_path):
    """Check if a trained model already exists"""
    return os.path.exists(model_path)


def main():
    """Main function to run the program with user choices"""
    model_path = 'C:/machine learning/drug_repurposing/models/drug_repurposing_predictor.pkl'
    drug_props_path = 'C:/machine learning/new/data/processed/drug_properties_realistic.csv'
    medicine_data_path = 'C:/machine learning/new/data/processed/enhanced_medicine_data.csv'
    
    # Create necessary directories
    os.makedirs('C:/machine learning/drug_repurposing/models', exist_ok=True)
    
    print("Drug Repurposing Predictor")
    print("==========================")
    
    # Check if model exists
    model_exists = check_model_exists(model_path)
    
    if model_exists:
        print(f"Found existing model at {model_path}")
        while True:
            choice = input("\nOptions:\n1. Use existing model\n2. Train new model\n3. Evaluate model\n4. Exit\nEnter choice (1-4): ")
            
            if choice == '1':
                # Load existing model
                try:
                    print("Loading model...")
                    model = DrugRepurposingPredictor.load_model(model_path)
                    print("Model loaded successfully!")
                    predict_with_model(model)
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Please train a new model.")
            
            elif choice == '2':
                # Train new model
                print("Training new neural network model...")
                model = DrugRepurposingPredictor()
                _ = model.train(drug_props_path, medicine_data_path, verbose=True, track_progress=True)
                model.save_model(model_path)
                
                # Ask if user wants to make predictions
                if input("Do you want to make predictions with the new model? (y/n): ").lower() == 'y':
                    predict_with_model(model)
            
            elif choice == '3':
                # Evaluate model
                try:
                    print("Loading model for evaluation...")
                    model = DrugRepurposingPredictor.load_model(model_path)
                    model.evaluate(drug_props_path, medicine_data_path)
                except Exception as e:
                    print(f"Error evaluating model: {e}")
            
            elif choice == '4':
                print("Exiting program.")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
    else:
        print("No existing model found.")
        print("Training new neural network model...")
        model = DrugRepurposingPredictor()
        _ = model.train(drug_props_path, medicine_data_path, verbose=True, track_progress=True)
        model.save_model(model_path)
        
        # Ask if user wants to make predictions
        if input("Do you want to make predictions with the new model? (y/n): ").lower() == 'y':
            predict_with_model(model)


def predict_with_model(model):
    """Function to make predictions with a trained model"""
    print("\nDrug Repurposing Prediction")
    print("==========================")
    
    while True:
        # Get drug information
        drug_name = input("\nEnter drug name (or 'q' to quit): ")
        if drug_name.lower() == 'q':
            break
            
        chemical_structure = input("Enter SMILES chemical structure (or press Enter for example): ")
        if not chemical_structure:
            chemical_structure = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example SMILES for aspirin
            print(f"Using example SMILES: {chemical_structure}")
        
        # Get molecular features with default values
        print("\nEnter molecular features (press Enter for default values):")
        try:
            mol_weight = float(input("Molecular weight [300]: ") or 300)
            h_donors = int(input("H-donors [2]: ") or 2)
            h_acceptors = int(input("H-acceptors [4]: ") or 4)
            rot_bonds = int(input("Rotatable bonds [3]: ") or 3)
            logp = float(input("LogP [2.5]: ") or 2.5)
        except ValueError:
            print("Invalid input. Using default values.")
            mol_weight, h_donors, h_acceptors, rot_bonds, logp = 300, 2, 4, 3, 2.5
        
        # Create molecular features dictionary
        molecular_features = {
            'molecular_weight': mol_weight,
            'h_donors': h_donors,
            'h_acceptors': h_acceptors,
            'rotatable_bonds': rot_bonds,
            'logp': logp
        }
        
        # Get prediction parameters
        try:
            confidence = float(input("\nConfidence threshold (0.0-1.0) [0.2]: ") or 0.2)
            top_n = int(input("Number of top predictions to show [5]: ") or 5)
        except ValueError:
            print("Invalid input. Using default values.")
            confidence, top_n = 0.2, 5
        
        # Make predictions
        print("\nPredicting potential therapeutic uses...")
        predictions = model.predict_repurposing(
            drug_name, 
            chemical_structure, 
            molecular_features,
            confidence_threshold=confidence,
            top_n=top_n
        )
        
        # Display results
        print("\nPredicted Therapeutic Uses:")
        print("--------------------------")
        for i, (condition, probability) in enumerate(predictions, 1):
            print(f"{i}. {condition} (Confidence: {probability:.2%})")
        
        # Ask if user wants to continue
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break


if __name__ == "__main__":
    main()