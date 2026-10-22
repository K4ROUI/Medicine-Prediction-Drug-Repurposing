import pandas as pd
import numpy as np
import pickle
import re
import time
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from tqdm import tqdm

# Adding Neural Network imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress TensorFlow/oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class MedicineSideEffectPredictor:
    def __init__(self):
        """
        Initialize the MedicineSideEffectPredictor with Neural Network model
        """
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model_type = 'nn'  # Always use neural network
        self.classifier = None  # Will be built during training
        self.mlb = MultiLabelBinarizer()
        self.training_history = {
            'accuracy': [],
            'hamming_loss': [],
            'iterations': []
        }
        
    def preprocess_side_effects(self, side_effects_text):
        """Extract individual side effects from the side effects text"""
        if pd.isna(side_effects_text):
            return []
            
        # Split by major delimiters but preserve phrases
        side_effects = re.split(r', |\s+and\s+|\n', side_effects_text)
        
        # Clean up each side effect but keep multi-word phrases intact
        side_effects = [re.sub(r'\s+', ' ', effect).strip().lower() for effect in side_effects if effect.strip()]
        
        # Remove any duplicates while preserving order
        seen = set()
        side_effects = [x for x in side_effects if not (x in seen or seen.add(x))]
        
        return side_effects
        
    def _build_neural_network(self, input_dim, output_dim):
        """Build a neural network model for multi-label classification"""
        # Clear any existing Keras session
        tf.keras.backend.clear_session()
        
        # Build a simple neural network for multi-label classification
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='sigmoid')
        ])
        
        # Compile model with binary crossentropy for multi-label classification
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, data_path, verbose=False, track_progress=True):
        """Train the model on the medicine data with progress tracking"""
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Create features from medicine name, composition, and uses
        df['features'] = df['medicine_name'] + ' ' + df['composition'] + ' ' + df['uses']
        
        # Extract and prepare the labels (side effects)
        print("Preparing features and labels...")
        side_effects_list = df['side_effects'].apply(self.preprocess_side_effects)
        
        # Transform labels to binary format
        y = self.mlb.fit_transform(side_effects_list)
        
        # Transform text features
        X = self.vectorizer.fit_transform(df['features'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reset training history
        self.training_history = {
            'accuracy': [],
            'hamming_loss': [],
            'iterations': []
        }
        
        # Convert sparse matrix to dense for Keras
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        # Build neural network model
        input_dim = X_train_dense.shape[1]
        output_dim = y_train.shape[1]
        
        print(f"Building Neural Network model with input dim: {input_dim}, output dim: {output_dim}")
        self.classifier = self._build_neural_network(input_dim, output_dim)
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        )
        
        # Train model with progress bar
        epochs = 20
        batch_size = 32
        
        if track_progress:
            # Create custom callback to track hamming loss during training
            class HammingLossCallback(tf.keras.callbacks.Callback):
                def __init__(self, X_test, y_test, training_history):
                    super().__init__()
                    self.X_test = X_test
                    self.y_test = y_test
                    self.training_history = training_history
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Make predictions
                    y_pred = (self.model.predict(self.X_test, verbose=0) > 0.5).astype(int)
                    
                    # Calculate hamming loss
                    h_loss = hamming_loss(self.y_test, y_pred)
                    
                    # Update training history
                    self.training_history['accuracy'].append(logs.get('val_accuracy'))
                    self.training_history['hamming_loss'].append(h_loss)
                    self.training_history['iterations'].append(epoch + 1)
                    
                    if verbose:
                        print(f"Epoch {epoch+1}: Val Accuracy: {logs.get('val_accuracy'):.4f}, Hamming Loss: {h_loss:.6f}")
            
            # Initialize and use the custom callback
            hamming_callback = HammingLossCallback(X_test_dense, y_test, self.training_history)
            
            # Train with history recording
            history = self.classifier.fit(
                X_train_dense, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_dense, y_test),
                callbacks=[early_stopping, hamming_callback],
                verbose=1 if verbose else 0
            )
        else:
            # Train without detailed history
            self.classifier.fit(
                X_train_dense, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test_dense, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
        
        print("Training complete!")
        
        # Calculate final metrics
        y_pred = (self.classifier.predict(X_test_dense, verbose=0) > 0.5).astype(int)
        
        metrics = {
            'subset_accuracy': accuracy_score(y_test, y_pred),
            'hamming_loss': hamming_loss(y_test, y_pred),
            'classes': self.mlb.classes_
        }
        
        # Add final point to history if needed
        if track_progress and len(self.training_history['iterations']) < epochs:
            self.training_history['accuracy'].append(metrics['subset_accuracy'])
            self.training_history['hamming_loss'].append(metrics['hamming_loss'])
            self.training_history['iterations'].append(len(self.training_history['iterations']) + 1)
        
        if verbose:
            print("\nModel Evaluation:")
            print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
            print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"Number of unique side effects: {len(self.mlb.classes_)}")
        
        return metrics
    
    def predict_side_effects(self, medicine_name, composition, uses, max_predictions=4):
        """
        Predict side effects for a new medicine
        
        Parameters:
        -----------
        medicine_name : str
            Name of the medicine
        composition : str
            Composition of the medicine
        uses : str
            Uses of the medicine
        max_predictions : int, default=4
            Maximum number of side effects to predict
            
        Returns:
        --------
        list
            List of individual predicted side effects
        """
        # Combine features
        features = medicine_name + ' ' + composition + ' ' + uses
        
        # Transform input
        X = self.vectorizer.transform([features])
        
        # Neural Network prediction
        pred_proba = self.classifier.predict(X.toarray(), verbose=0)[0]
        
        # Get indices sorted by probability (highest first)
        sorted_indices = np.argsort(pred_proba)[::-1]
        
        # Extract individual side effects
        all_side_effects = []
        seen = set()  # Track seen side effects to avoid duplicates
        
        # Continue until we have the required number of unique side effects
        i = 0
        while len(all_side_effects) < max_predictions and i < len(sorted_indices):
            idx = sorted_indices[i]
            i += 1
            
            # Skip if probability is too low
            if pred_proba[idx] < 0.1:  # Only consider predictions with reasonable confidence
                continue
                
            # Get the side effect string and split into individual effects
            effect_string = self.mlb.classes_[idx]
            
            # Split multi-word effects carefully
            individual_effects = re.split(r', |\s+and\s+|\n', effect_string)
            
            for effect in individual_effects:
                effect = effect.strip().lower()
                if not effect or effect in seen:
                    continue
                
                seen.add(effect)
                all_side_effects.append(effect.capitalize())  # Capitalize for better presentation
                
                # Break if we've reached the max number
                if len(all_side_effects) >= max_predictions:
                    break
        
        return all_side_effects
    
    def save_model(self, path):
        """Save the trained model to disk"""
        # For neural network, save the model separately with proper extension
        model_path = path.replace('.pkl', '_nn_model.keras')
        self.classifier.save(model_path)
        
        # Save everything except the classifier
        model_data = {
            'vectorizer': self.vectorizer,
            'mlb': self.mlb,
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
        model = MedicineSideEffectPredictor()
        
        # Load components
        model.vectorizer = model_data['vectorizer']
        model.mlb = model_data['mlb']
        
        if 'training_history' in model_data:
            model.training_history = model_data['training_history']
            
        # Load neural network model
        nn_path = model_data.get('nn_model_path')
        if nn_path:
            model.classifier = tf.keras.models.load_model(nn_path)
        else:
            raise ValueError("Neural network model path not found in saved data")
        
        return model
    
    def evaluate(self, data_path):
        """Evaluate the model on the given dataset with progress visualization"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Create features from medicine name, composition, and uses
        df['features'] = df['medicine_name'] + ' ' + df['composition'] + ' ' + df['uses']
        
        # Extract and prepare the labels (side effects)
        side_effects_list = df['side_effects'].apply(self.preprocess_side_effects)
        
        # Transform labels to binary format
        y_true = self.mlb.transform(side_effects_list)
        
        # Transform text features
        X = self.vectorizer.transform(df['features'])
        
        # Make predictions
        print("Evaluating model...")
        with tqdm(total=100, desc="Evaluation Progress") as pbar:
            pbar.update(50)
            
            # Neural network prediction
            y_pred = (self.classifier.predict(X.toarray(), verbose=0) > 0.5).astype(int)
                
            pbar.update(50)
        
        # Calculate metrics
        subset_accuracy = accuracy_score(y_true, y_pred)
        h_loss = hamming_loss(y_true, y_pred)
        
        # Create and display graphs showing the model progress and final metrics
        self._plot_evaluation_with_progress(subset_accuracy, h_loss)
        
        # Additional analysis - summary only
        print("\nOverall metrics:")
        print(f"Subset Accuracy: {subset_accuracy:.4f}")
        print(f"Hamming Loss: {h_loss:.4f}")
        
        return {
            'subset_accuracy': subset_accuracy,
            'hamming_loss': h_loss
        }
    
    def _plot_evaluation_with_progress(self, final_accuracy, final_loss):
        """Plot model evaluation metrics and training progress"""
        try:
            # Create directory for results if it doesn't exist
            os.makedirs('C:/machine learning/new/results/figures', exist_ok=True)
            
            # Create figure with 2 rows, 1 column layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot accuracy progress
            if self.training_history['iterations']:
                ax1.plot(self.training_history['iterations'], self.training_history['accuracy'], 
                         marker='o', linestyle='-', color='green', label='Training Progress')
                ax1.axhline(y=final_accuracy, color='blue', linestyle='--', label=f'Final: {final_accuracy:.4f}')
                ax1.set_title('Accuracy Progress During Neural Network Model Training')
                ax1.set_xlabel('Training Epochs')
                ax1.set_ylabel('Accuracy')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()
                
                # Calculate appropriate y-axis limits for Hamming Loss
                min_loss = min(self.training_history['hamming_loss'])
                max_loss = max(self.training_history['hamming_loss'])
                
                # Add some padding to the limits (10% of the range)
                loss_range = max_loss - min_loss
                if loss_range > 0:
                    padding = 0.1 * loss_range
                    y_min = max(0, min_loss - padding)  # Ensure it doesn't go below 0
                    y_max = max_loss + padding
                else:
                    # If range is very small, create an artificial range around the value
                    base_value = min_loss if min_loss > 0 else 0.001
                    y_min = base_value * 0.8
                    y_max = base_value * 1.2
                
                # Hamming loss progress with adjusted y-axis
                ax2.plot(self.training_history['iterations'], self.training_history['hamming_loss'], 
                         marker='o', linestyle='-', color='red', label='Training Progress')
                ax2.axhline(y=final_loss, color='blue', linestyle='--', label=f'Final: {final_loss:.6f}')
                ax2.set_title('Hamming Loss Progress During Neural Network Model Training')
                ax2.set_xlabel('Training Epochs')
                ax2.set_ylabel('Hamming Loss (Lower is Better)')
                ax2.set_ylim(y_min, y_max)  # Set the calculated limits
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()
                
                # Add annotations for significant changes
                for i, (epoch, loss) in enumerate(zip(self.training_history['iterations'], self.training_history['hamming_loss'])):
                    if i > 0 and i < len(self.training_history['iterations'])-1:
                        prev_loss = self.training_history['hamming_loss'][i-1]
                        if abs(loss - prev_loss) > 0.00001:  # Threshold for "significant" change
                            ax2.annotate(f'{loss:.6f}', 
                                         xy=(epoch, loss),
                                         xytext=(0, 10),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=8,
                                         arrowprops=dict(arrowstyle='->', lw=0.5))
            else:
                # No progress data available
                ax1.text(0.5, 0.5, "No training progress data available", 
                         ha='center', va='center')
                ax2.text(0.5, 0.5, "Train with track_progress=True to see progress", 
                         ha='center', va='center')
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig('C:/machine learning/new/results/figures/side_effect_model_evaluation_progress_nn.png')
            print(f"Evaluation graph saved to C:/machine learning/new/results/figures/side_effect_model_evaluation_progress_nn.png")
            
            # Show the plot
            plt.show()
        except Exception as e:
            print(f"Could not create evaluation graph: {e}")
            print("Please install matplotlib with 'pip install matplotlib' to enable graphs.")

    def clear_history(self):
        """Clear the training history data"""
        self.training_history = {
            'accuracy': [],
            'hamming_loss': [],
            'iterations': []
        }


def check_model_exists(model_path):
    """Check if a trained model already exists"""
    return os.path.exists(model_path)


def main():
    """Main function to run the program with user choices"""
    model_path = 'C:/machine learning/new/models/medicine_side_effect_predictor.pkl'
    data_path = 'C:/machine learning/new/data/processed/enhanced_medicine_data.csv'
    
    # Create necessary directories
    os.makedirs('C:/machine learning/new/models', exist_ok=True)
    os.makedirs('C:/machine learning/new/results/figures', exist_ok=True)
    
    # Check if model exists
    model_exists = check_model_exists(model_path)
    model = None
    
    while True:
        print("\n=== Medicine Side Effect Predictor ===")
        print("1. Train model")
        print("2. Evaluate model")
        print("3. Predict medicine side effects")
        print("4. Exit")
        
        if model_exists:
            print("\nA trained model already exists!")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            if model_exists:
                retrain = input("Model already exists. Retrain? (y/n): ").lower()
                if retrain != 'y':
                    continue
            
            # Train new model
            model = MedicineSideEffectPredictor()
            print("\nTraining neural network model...")
            metrics = model.train(data_path, verbose=True, track_progress=True)
            model.save_model(model_path)
            model_exists = True
            print(f"\nModel trained successfully with accuracy: {metrics['subset_accuracy']:.4f}")
            
        elif choice == '2':
            if not model_exists:
                print("No trained model found. Please train a model first.")
                continue
                
            # Load model if not already loaded
            if model is None:
                print("Loading model...")
                model = MedicineSideEffectPredictor.load_model(model_path)
                
            # Evaluate model
            print("\nEvaluating model...")
            metrics = model.evaluate(data_path)
            
        elif choice == '3':
            if not model_exists:
                print("No trained model found. Please train a model first.")
                continue
                
            # Load model if not already loaded
            if model is None:
                print("Loading model...")
                model = MedicineSideEffectPredictor.load_model(model_path)
            
            # Get input from user
            print("\nEnter medicine details:")
            medicine_name = input("Medicine Name: ")
            composition = input("Composition: ")
            uses = input("Uses: ")
            
            # Allow user to specify max number of predictions
            max_predictions = 4  # Default is 4
            try:
                user_max = input(f"Max number of side effects to show (default: {max_predictions}): ")
                if user_max.strip():
                    max_predictions = int(user_max)
            except ValueError:
                print(f"Invalid input. Using default ({max_predictions}).")
            
            # Make prediction with limited number of side effects
            predicted_side_effects = model.predict_side_effects(
                medicine_name, composition, uses, max_predictions=max_predictions
            )
            
            print("\nPredicted side effects:")
            if predicted_side_effects:
                for i, effect in enumerate(predicted_side_effects, 1):
                    print(f"{i}. {effect}")
            else:
                print("No specific side effects predicted for this medicine.")
            
        elif choice == '4':
            print("Exiting program.")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()