import os
import sys
import logging
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, flash, jsonify, Response
import time
from flask import make_response
from fuzzywuzzy import process, fuzz
import pandas as pd
import gc
import signal
from functools import wraps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import _thread
import atexit
import platform
import threading
from transformers import pipeline

# Resolve all runtime assets from the repository instead of a developer's
# local machine. This keeps the application portable after cloning.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the translation model only when translation is requested. Explicitly
# selecting PyTorch avoids Transformers probing the incompatible TensorFlow/Keras
# backend during application startup.
translator = None

def translate_outputs(outputs):
    global translator
    if translator is None:
        translator = pipeline(
            "translation_en_to_fr",
            model="Helsinki-NLP/opus-mt-en-fr",
            framework="pt",
        )
    return [translator(output, max_length=60)[0]['translation_text'] for output in outputs]

# Add the DrugSearchEngine class
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
                logging.info(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.medicine_dict[best_match]
        logging.info(f"Medicine '{query}' not found")
        return None
    
    def search_drug(self, query, fuzzy_match=True, threshold=80):
        query = query.lower().strip()
        if query in self.drug_dict:
            return self.drug_dict[query]
        if fuzzy_match and self.drug_dict:
            matches = process.extractOne(query, self.drug_dict.keys(), scorer=fuzz.token_sort_ratio)
            if matches and matches[1] >= threshold:
                best_match = matches[0]
                logging.info(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
                return self.search_drug(best_match, fuzzy_match=False)
        logging.info(f"Drug '{query}' not found")
        return None
    
    def get_drugs_in_medicine(self, medicine_name):
        medicine_name = medicine_name.lower().strip()
        if medicine_name in self.medicine_to_drugs:
            return self.medicine_to_drugs[medicine_name]
        matches = process.extractOne(medicine_name, self.medicine_to_drugs.keys(), scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            logging.info(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.medicine_to_drugs[best_match]
        logging.info(f"Medicine '{medicine_name}' not found")
        return []
    
    def get_medicines_containing_drug(self, drug_name):
        drug_name = drug_name.lower().strip()
        if drug_name in self.drug_to_medicines:
            return self.drug_to_medicines[drug_name]
        matches = process.extractOne(drug_name, self.drug_to_medicines.keys(), scorer=fuzz.token_sort_ratio)
        if matches and matches[1] >= 80:
            best_match = matches[0]
            logging.info(f"Found close match: '{best_match}' (similarity: {matches[1]}%)")
            return self.drug_to_medicines[best_match]
        logging.info(f"Drug '{drug_name}' not found in any medicine")
        return []
    
    def search_by_side_effect(self, side_effect_query):
        if self.med_data is None or 'side_effects' not in self.med_data.columns:
            logging.warning("Side effects data not available")
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to Python path so we can import 'src'
sys.path.insert(0, PROJECT_ROOT)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'development-only-change-me')
app.jinja_env.globals.update(min=min)
# Model paths (absolute)
MODEL_PATHS = {
    'use': os.path.join(PROJECT_ROOT, 'models', 'medicine_use_predictor.pkl'),
    'side_effect': os.path.join(PROJECT_ROOT, 'models', 'medicine_side_effect_predictor.pkl'),
    'repurposing': os.path.join(PROJECT_ROOT, 'models', 'drug_repurposing_predictor.pkl')
}

# Data paths
DATA_PATHS = {
    'use': os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv'),
    'side_effect': os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv'),
    'repurposing': {
        'drug_props': os.path.join(PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv'),
        'medicine_data': os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv')
    },
    'search_engine': {
        'medicine_data': os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv'),
        'drug_props': os.path.join(PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv'),
        'repurposing_data': os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_repurposing_data.csv')
    }
}

# Existing graph paths
GRAPH_PATHS = {
    'use': os.path.join(PROJECT_ROOT, 'results', 'figures', 'model_evaluation_progress_nn.png'),
    'side_effect': os.path.join(PROJECT_ROOT, 'results', 'figures', 'side_effect_model_evaluation_progress_nn.png'),
    'repurposing': os.path.join(PROJECT_ROOT, 'results', 'figures', 'model_evaluation_progress_nn1.png')
}

# Store loaded models to prevent reloading issues
loaded_models = {}
# Create a shared instance of the search engine
search_engine = None

# Initialize the search engine
def initialize_search_engine():
    global search_engine
    try:
        search_engine = DrugSearchEngine(
            medicine_data_path=DATA_PATHS['search_engine']['medicine_data'],
            drug_properties_path=DATA_PATHS['search_engine']['drug_props'],
            repurposing_data_path=DATA_PATHS['search_engine']['repurposing_data']
        )
        logging.info("Search engine initialized successfully")
        stats = search_engine.get_stats()
        logging.info(f"Database loaded with {stats['num_medicines']} medicines and {stats['num_unique_drugs']} drugs")
        return True
    except Exception as e:
        logging.error(f"Error initializing search engine: {e}", exc_info=True)
        return False

# Ensure model directories exist
def ensure_directories():
    for path in MODEL_PATHS.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            logging.warning(f"Model file does not exist: {path}")
    
    # Also ensure results directory exists for evaluation figures
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Utility to check existence
def check_model_exists(path):
    if not os.path.exists(path):
        logging.warning(f"Model not found: {path}")
        return False
    return True

def load_model(model_type, force_reload=False):
    # Return cached model if available and not forcing reload
    if model_type in loaded_models and not force_reload:
        logging.info(f"Using cached {model_type} model")
        return loaded_models[model_type]
    
    # Always clear matplotlib figures before doing anything
    plt.close('all')
    
    # Clear the model from cache regardless if we're forcing a reload
    if model_type in loaded_models:
        logging.info(f"Clearing {model_type} model from cache")
        del loaded_models[model_type]
        gc.collect()  # Force garbage collection
    
    path = MODEL_PATHS.get(model_type)
    if not path or not check_model_exists(path):
        logging.error(f"Model '{model_type}' missing or incorrect path: {path}")
        return None
    
    try:
        # Close all matplotlib figures before loading a new model again to be safe
        plt.close('all')
        
        if model_type == 'use':
            from src.models.medicine_use_predictor import MedicineUsePredictor
            model = MedicineUsePredictor.load_model(path)
        elif model_type == 'side_effect':
            from src.models.medicine_side_effect_predictor import MedicineSideEffectPredictor
            model = MedicineSideEffectPredictor.load_model(path)
        elif model_type == 'repurposing':
            # Import the specific model class directly - avoid relative imports
            import pickle
            import os
            import sys
            
            # Clear any TensorFlow session state if TensorFlow is being used
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                logging.info("TensorFlow not available, skipping session cleanup")
            
            # More reliable error handling for repurposing model
            try:
                from src.models.drug_repurposing_predictor import DrugRepurposingPredictor
                model = DrugRepurposingPredictor.load_model(path)
            except (ImportError, Exception) as e:
                logging.error(f"Failed to load repurposing model: {e}")
                return None
                
            # Verify the model has necessary components
            if not hasattr(model, 'text_vectorizer') or not hasattr(model, 'classifier'):
                logging.error("Loaded model is missing required components")
                return None
        else:
            logging.error(f"Unknown model type '{model_type}'")
            return None
        
        logging.info(f"Successfully loaded {model_type} model from {path}")
        if hasattr(model, 'reset_state') and callable(model.reset_state):
            model.reset_state()
            
        # Cache the loaded model
        loaded_models[model_type] = model
        return model
    except Exception as e:
        logging.error(f"Error loading '{model_type}' model: {e}", exc_info=True)
        # Clean up resources on error
        plt.close('all')
        gc.collect()
        return None

# Function to use existing graphs instead of generating new ones
def get_evaluation_plots(metrics, model_type):
    plots = {}
    
    # Check if metrics exist (they're still needed for the evaluation data)
    if not metrics:
        return plots
    
    # Get the path to the existing graph for this model type
    graph_path = GRAPH_PATHS.get(model_type)
    
    if not graph_path or not os.path.exists(graph_path):
        logging.warning(f"Graph not found for {model_type} at path: {graph_path}")
        flash(f"Pre-existing graph for {model_type} not found", "warning")
        return plots
        
    try:
        # Load the existing graph as an image and convert to base64
        with open(graph_path, 'rb') as img_file:
            img_data = img_file.read()
            plots['training_progress'] = base64.b64encode(img_data).decode('utf-8')
            logging.info(f"Successfully loaded existing graph for {model_type} from {graph_path}")
    except Exception as e:
        logging.error(f"Error loading graph file: {e}")
        flash(f"Error loading graph: {e}", "error")
    
    # Still include any metrics data for display in the template
    plots['metrics'] = metrics
    
    return plots

# Routes
@app.route('/')
def index():
    status = {k: check_model_exists(v) for k, v in MODEL_PATHS.items()}
    graph_status = {k: os.path.exists(v) for k, v in GRAPH_PATHS.items()}
    
    # Add search engine status
    search_status = search_engine is not None
    
    return render_template('index.html', 
                           model_status=status, 
                           graph_status=graph_status,
                           search_status=search_status)
def timeout_handler(func, args=(), kwargs={}, timeout_duration=10):
    """
    Windows-compatible function timeout handler using threading
    """
    result = [None]
    error = [None]
    finished = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
        finally:
            finished[0] = True
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if not finished[0]:
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_duration} seconds")
    if error[0]:
        raise error[0]
    return result[0]

@app.route('/medicine_use', methods=['GET', 'POST'])
def medicine_use():
    predicted_uses = None
    translated_uses = None
    name = ''
    comp = ''
    
    if request.method == 'POST':
        name = request.form.get('medicine_name', '').strip()
        comp = request.form.get('composition', '').strip()
        try:
            model = load_model('use')
            if model:
                try:
                    # Use timeout handler - 10 second timeout
                    predicted_uses = timeout_handler(model.predict_use, args=(name, comp), timeout_duration=10)
                    
                    # Translate uses to French if prediction succeeded
                    if predicted_uses:
                        try:
                            translated_uses = translate_outputs(predicted_uses)
                        except Exception as e:
                            flash(f"Translation failed: {e}", "warning")
                            logging.error(f"Translation error: {e}", exc_info=True)
                except Exception as e:
                    flash(f"Prediction error: {e}", "error")
                    logging.error(f"Prediction error: {e}", exc_info=True)
            else:
                flash("Failed to load model - please try again", "error")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "error")
            logging.error(f"Unexpected error in medicine_use: {e}", exc_info=True)
        finally:
            plt.close('all')
            gc.collect()
    
    return render_template(
    'medicine_use.html',
    predicted_uses=predicted_uses,
    translated_uses=translated_uses,
    medicine_name=name,
    composition=comp
)

@app.route('/side_effect', methods=['GET', 'POST'])
def side_effect():
    preds = None
    name = comp = uses = ''
    maxp = 1
    if request.method == 'POST':
        name = request.form.get('medicine_name', '').strip()
        comp = request.form.get('composition', '').strip()
        uses = request.form.get('uses', '').strip()
        try:
            maxp = int(request.form.get('max_predictions', 4))
        except ValueError:
            flash("Invalid max_predictions; defaulting to 4.", "warning")
        model = load_model('side_effect')
        if model:
            try:
                preds = model.predict_side_effects(name, comp, uses, max_predictions=maxp)
            except Exception as e:
                flash(f"Prediction error: {e}", "error")
    return render_template('side_effect.html', predicted_side_effects=preds, medicine_name=name, composition=comp, uses=uses, max_predictions=maxp)

@app.route('/repurposing', methods=['GET', 'POST'])
def repurposing():
    preds = None
    drug = ''
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    feats = {'molecular_weight':300.0,'h_donors':2,'h_acceptors':4,'rotatable_bonds':3,'logp':2.5}
    conf, topn = 0.2, 5
    if request.method == 'POST':
        drug = request.form.get('drug_name','').strip()
        smiles = request.form.get('chemical_structure','').strip() or smiles
        try:
            feats['molecular_weight'] = float(request.form.get('mol_weight', feats['molecular_weight']))
            feats['h_donors'] = int(request.form.get('h_donors', feats['h_donors']))
            feats['h_acceptors'] = int(request.form.get('h_acceptors', feats['h_acceptors']))
            feats['rotatable_bonds'] = int(request.form.get('rot_bonds', feats['rotatable_bonds']))
            feats['logp'] = float(request.form.get('logp', feats['logp']))
            conf = float(request.form.get('confidence', conf))
            topn = int(request.form.get('top_n', topn))
        except ValueError as e:
            flash(f"Invalid input: {e}", "error")
        
        # Additional debugging for repurposing
        logging.info(f"Loading repurposing model from {MODEL_PATHS['repurposing']}")
        model = load_model('repurposing')
        
        if model:
            try:
                logging.info("Attempting prediction with repurposing model")
                preds = model.predict_repurposing(drug, smiles, feats, confidence_threshold=conf, top_n=topn)
                logging.info(f"Successfully made prediction with {len(preds)} results")
            except Exception as e:
                logging.error(f"Detailed prediction error: {str(e)}", exc_info=True)
                flash(f"Prediction error: {e}", "error")
    return render_template('repurposing.html', predictions=preds, drug_name=drug, chemical_structure=smiles, molecular_features=feats, confidence=conf, top_n=topn)

# Create a timeout decorator to prevent any function from running forever
def timeout(seconds=60, error_message="Function call timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout handler
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the timeout alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# Add this function to help kill hanging evaluation processes
def force_terminate_thread():
    """Force terminate the current thread - use only as a last resort"""
    _thread.interrupt_main()


@app.route('/evaluate_model/<model_type>', methods=['GET', 'POST'])
def evaluate_model(model_type):
    if model_type not in MODEL_PATHS:
        flash(f"Unknown model type: {model_type}", "error")
        return render_template('evaluate_model.html', model_type=model_type, metrics=None, plots=None)
    
    metrics = None
    plots = None
    
    # Clear any existing matplotlib figures to avoid memory leaks
    plt.close('all')
    
    if request.method == 'POST':
        try:
            # Create a tracker file to indicate evaluation is in progress
            # Use a temp directory that works on Windows
            import tempfile
            temp_dir = tempfile.gettempdir()
            tracker_file = os.path.join(temp_dir, f"model_eval_{model_type}.lock")
            
            with open(tracker_file, 'w') as f:
                f.write(str(time.time()))
            
            # First, make sure we clean up any existing model instances
            if model_type in loaded_models:
                logging.info(f"Removing cached {model_type} model")
                del loaded_models[model_type]
                
            # Close all matplotlib figures again
            plt.close('all')
            
            # Force garbage collection
            gc.collect()
            
            # Load the model with forced reload
            logging.info(f"Loading {model_type} model with force_reload=True")
            model = load_model(model_type, force_reload=True)
            
            if not model:
                flash(f"Failed to load {model_type} model", "error")
                if os.path.exists(tracker_file):
                    os.remove(tracker_file)
            else:
                # Set a timeout for evaluation
                evaluation_timeout = 120  # 2 minutes timeout
                
                try:
                    # For Windows compatibility, use a simpler timeout approach
                    logging.info(f"Starting evaluation of {model_type} model")
                    
                    start_time = time.time()
                    
                    # Run the evaluation
                    if model_type == 'repurposing':
                        # Special case for repurposing model which needs two datasets
                        drug_props_path = DATA_PATHS['repurposing']['drug_props']
                        medicine_data_path = DATA_PATHS['repurposing']['medicine_data']
                        metrics = model.evaluate(drug_props_path, medicine_data_path)
                    else:
                        # Standard case for other models
                        data_path = DATA_PATHS.get(model_type)
                        metrics = model.evaluate(data_path)
                    
                    elapsed_time = time.time() - start_time
                    logging.info(f"Evaluation completed for {model_type} model in {elapsed_time:.2f} seconds")
                    
                    if metrics:
                        # Use existing graphs instead of generating new ones
                        plots = get_evaluation_plots(metrics, model_type)
                        flash(f"{model_type.capitalize()} model evaluation completed successfully in {elapsed_time:.2f} seconds", "success")
                    else:
                        flash(f"Model evaluation didn't return any metrics", "warning")
                
                except Exception as e:
                    logging.error(f"Error during evaluation process: {str(e)}", exc_info=True)
                    flash(f"Evaluation error: {e}", "error")
                finally:
                    # Make sure to clean up
                    if model_type in loaded_models:
                        del loaded_models[model_type]
                    gc.collect()
                    plt.close('all')
                    if os.path.exists(tracker_file):
                        os.remove(tracker_file)
        
        except Exception as e:
            logging.error(f"Unexpected error in evaluation route: {str(e)}", exc_info=True)
            flash(f"Unexpected error: {e}", "error")
            # Make sure to close all matplotlib figures on error
            plt.close('all')
            gc.collect()
    
    # Add a cache-busting parameter to prevent browser caching
    response = make_response(render_template('evaluate_model.html', 
                                           model_type=model_type, 
                                           metrics=metrics, 
                                           plots=plots,
                                           cache_buster=int(time.time())))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# New route for drug search
@app.route('/drug_search', methods=['GET', 'POST'])
def drug_search():
    search_type = 'drug'  # Default search type
    query = ''
    results = None
    error = None
    
    # Check if search engine is initialized
    if search_engine is None:
        if not initialize_search_engine():
            flash("Drug search engine could not be initialized", "error")
            return render_template('drug_search.html')
    
    if request.method == 'POST':
        search_type = request.form.get('search_type', 'drug')
        query = request.form.get('search_query', '').strip()
        
        if not query:
            flash("Please enter a search query", "warning")
        else:
            try:
                if search_type == 'drug':
                    results = search_engine.search_drug(query)
                elif search_type == 'medicine':
                    results = search_engine.search_medicine(query)
                elif search_type == 'side_effect':
                    results = search_engine.search_by_side_effect(query)
                elif search_type == 'medicines_with_drug':
                    results = search_engine.get_medicines_containing_drug(query)
                elif search_type == 'drugs_in_medicine':
                    results = search_engine.get_drugs_in_medicine(query)
                
                if not results and search_type not in ['side_effect', 'medicines_with_drug', 'drugs_in_medicine']:
                    flash(f"No results found for '{query}'", "info")
                elif not results and search_type in ['side_effect', 'medicines_with_drug', 'drugs_in_medicine'] and isinstance(results, list):
                    flash(f"No results found for '{query}'", "info")
                    
            except Exception as e:
                logging.error(f"Search error: {str(e)}", exc_info=True)
                flash(f"Error during search: {e}", "error")
                error = str(e)
    
    stats = search_engine.get_stats() if search_engine else None
    
    return render_template('drug_search.html',
                          search_type=search_type,
                          search_query=query,
                          results=results,
                          error=error,
                          stats=stats)
@app.route('/drug_clustering', methods=['GET', 'POST'])
def drug_clustering():
    results = None
    plots = {}
    error = None
    clustering_type = 'drugs'
    
    # Check if necessary files exist
    drug_file_exists = os.path.exists(DATA_PATHS['search_engine']['drug_props'])
    medicine_file_exists = os.path.exists(DATA_PATHS['search_engine']['medicine_data'])
    
    if not drug_file_exists or not medicine_file_exists:
        flash("Required data files not found", "error")
        return render_template('drug_clustering.html', 
                              results=None, 
                              plots={}, 
                              file_status={'drugs': drug_file_exists, 'medicines': medicine_file_exists})
    
    if request.method == 'POST':
        clustering_type = request.form.get('clustering_type', 'drugs')
        n_clusters = request.form.get('n_clusters', '15')
        
        try:
            n_clusters = int(n_clusters)
            if n_clusters < 2 or n_clusters > 30:
                flash("Number of clusters should be between 2 and 30", "warning")
                n_clusters = 15
        except ValueError:
            flash("Invalid number of clusters, using default value of 15", "warning")
            n_clusters = 15
            
        try:
            # Load data based on clustering type
            if clustering_type == 'drugs':
                df = pd.read_csv(DATA_PATHS['search_engine']['drug_props'])
                results = cluster_drugs(df, n_clusters)
                plots = generate_drug_plots(df, results, n_clusters)
            elif clustering_type == 'medicine_uses':
                df = pd.read_csv(DATA_PATHS['search_engine']['medicine_data'])
                results = cluster_medicines_by_uses(df, n_clusters)
                plots = generate_medicine_uses_plots(df, results, n_clusters)
            elif clustering_type == 'medicine_side_effects':
                df = pd.read_csv(DATA_PATHS['search_engine']['medicine_data'])
                results = cluster_medicines_by_side_effects(df, n_clusters)
                plots = generate_medicine_side_effects_plots(df, results, n_clusters)
                
            flash(f"Successfully clustered {clustering_type} into {n_clusters} groups", "success")
                
        except Exception as e:
            logging.error(f"Clustering error: {str(e)}", exc_info=True)
            flash(f"Error during clustering: {e}", "error")
            error = str(e)
            
    return render_template('drug_clustering.html',
                         clustering_type=clustering_type,
                         n_clusters=n_clusters if 'n_clusters' in locals() else 15,
                         results=results,
                         plots=plots,
                         error=error)

# Helper functions for the clustering route
def cluster_drugs(df, n_clusters=15):
    if 'drug' in df.columns:
        df = df.set_index('drug')

    features = ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']
    data = df[features].dropna()
    
    km = KMeans(n_clusters=n_clusters, random_state=42)
    data['colony'] = km.fit_predict(data)
    
    # Return the clustered data
    return data.reset_index()

def generate_drug_plots(df, clustered_data, n_clusters=15):
    plots = {}
    
    features = ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']
    data = df[features].dropna()
    
    # Use the clustered data from the previous function
    data['colony'] = clustered_data['colony']
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(data[features])
    
    # Generate scatter plot
    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        mask = data['colony'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Colony {c}', alpha=0.7)
    plt.title('Drug Colonies by Physicochemical Properties')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['scatter'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Generate bar chart
    counts = data['colony'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Drugs per Colony')
    plt.xlabel('Colony')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['bar'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots

def cluster_medicines_by_uses(df, n_clusters=15):
    uses = df['uses'].fillna('')
    
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(uses)
    
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df['uses_group'] = km.fit_predict(X)
    
    return df

def generate_medicine_uses_plots(df, clustered_data, n_clusters=15):
    plots = {}
    
    uses = df['uses'].fillna('')
    df['uses_group'] = clustered_data['uses_group']
    
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(uses)
    
    # Bar chart
    counts = df['uses_group'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Medicines per Uses-Cluster')
    plt.xlabel('Uses-Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['bar'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Scatter plot
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        mask = df['uses_group'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Group {c}', alpha=0.7)
    plt.title('Medicine Uses Clusters (TF-IDF + PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['scatter'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots

def cluster_medicines_by_side_effects(df, n_clusters=15):
    sefs = df['side_effects'].fillna('')
    
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(sefs)
    
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df['side_effects_group'] = km.fit_predict(X)
    
    return df

def generate_medicine_side_effects_plots(df, clustered_data, n_clusters=15):
    plots = {}
    
    sefs = df['side_effects'].fillna('')
    df['side_effects_group'] = clustered_data['side_effects_group']
    
    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(sefs)
    
    # Bar chart
    counts = df['side_effects_group'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Medicines per Side-Effects-Cluster')
    plt.xlabel('Side-Effects-Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['bar'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Scatter plot
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        mask = df['side_effects_group'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Group {c}', alpha=0.7)
    plt.title('Medicine Side Effects Clusters (TF-IDF + PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    
    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    plots['scatter'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plots
def cleanup_resources():
    """Clean up all resources on application exit"""
    logging.info("Cleaning up application resources...")
    plt.close('all')
    # Clear all loaded models
    loaded_models.clear()
    gc.collect()
    logging.info("Cleanup complete")

# Register the cleanup function to run on exit
atexit.register(cleanup_resources)

@app.errorhandler(404)
def not_found(e): return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e): return render_template('500.html'), 500

if __name__ == '__main__':
    ensure_directories()
    initialize_search_engine()
    # Increase the timeout and run with threaded=True
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000,
            use_reloader=False)  # use_reloader=False prevents double loading