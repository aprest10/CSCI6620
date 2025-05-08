from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.io import mmread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import re

from sklearn.metrics import (classification_report, f1_score, 
                            precision_score, recall_score, accuracy_score, confusion_matrix)
import seaborn as sns

class SparseMatrixPredictor:
    def __init__(self, optimal_formats_df=None):
        self.model = None
        self.feature_names = None
        self.class_names = None
        self.le = LabelEncoder()
        
        if optimal_formats_df is not None:
            self.train_model(optimal_formats_df)
    
    def prepare_features(self, optimal_formats_df):
        required_features = ['matrix_size', 'density', 'nnz', 'matrix_type']
        for feat in required_features:
            if feat not in optimal_formats_df.columns:
                raise ValueError(f"Missing required feature: {feat}")
        
        X = optimal_formats_df[['matrix_size', 'density', 'nnz']]
        matrix_type_dummies = pd.get_dummies(optimal_formats_df['matrix_type'], prefix='type')
        X = pd.concat([X, matrix_type_dummies], axis=1)
        
        if 'format' not in optimal_formats_df.columns:
            raise ValueError("Missing 'format' column in optimal formats data")
        
        y = self.le.fit_transform(optimal_formats_df['format'])
        class_names = self.le.classes_
        
        return X, y, X.columns.tolist(), class_names
    
    def train_model(self, optimal_formats_df, max_depth=4, min_samples_split=5, random_state=42, test_size=0.2):
        X, y, feature_names, class_names = self.prepare_features(optimal_formats_df)
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No data available for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Store model and metadata
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Calculate accuracy
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        return {
            "training_accuracy": train_acc,
            "testing_accuracy": test_acc,
            "feature_names": feature_names,
            "class_names": class_names.tolist()
        }
    
    def predict_format(self, matrix_size, density, nnz, matrix_type):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Create feature vector
        features = pd.DataFrame({
            'matrix_size': [matrix_size],
            'density': [density],
            'nnz': [nnz]
        })
        
        for col in self.feature_names:
            if col.startswith('type_'):
                features[col] = 1 if col == f'type_{matrix_type.lower()}' else 0
        
        features = features[self.feature_names]
        
        format_idx = self.model.predict(features)[0]
        return self.class_names[format_idx]

    def predict_formats_for_folder(self, folder_path):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        mtx_files = glob.glob(os.path.join(folder_path, "*.mtx"))
        if not mtx_files:
            raise ValueError(f"No .mtx files found in {folder_path}")
        
        results = []
        
        for file_path in mtx_files:
            try:
                filename = os.path.basename(file_path)
                
                try:
                    matrix = mmread(file_path)
                except Exception as e:
                    print(f"Could not read matrix file {filename}: {str(e)}")
                    continue
                
                rows, cols = matrix.shape
                nnz = matrix.nnz
                
                if rows == 0 or cols == 0:
                    print(f"Skipping empty matrix {filename}")
                    continue
                    
                density = nnz / (rows * cols)
                
                matrix_type = 'unknown'
                match = re.search(r'_([a-zA-Z]+)_', filename.lower())
                if match:
                    matrix_type = match.group(1).lower()
                
                predicted_format = self.predict_format(
                    matrix_size=max(rows, cols),
                    density=density,
                    nnz=nnz,
                    matrix_type=matrix_type
                )
                
                results.append({
                    'filename': filename,
                    'rows': rows,
                    'cols': cols,
                    'nnz': nnz,
                    'density': density,
                    'matrix_type': matrix_type,
                    'predicted_format': predicted_format
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No valid matrices processed successfully")
        
        return pd.DataFrame(results)
    
    def visualize_tree(self, save_path=None, figsize=(15, 10)):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        fig = plt.figure(figsize=figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("Decision Tree for Optimal Matrix Storage Format")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        return None
    
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        importances = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def print_decision_rules(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        print("\nDecision Rules:")
        feature_names = np.array(self.feature_names)
        
        def print_tree_rules(tree, node=0, depth=0):
            if tree.children_left[node] == tree.children_right[node]:  # Leaf node
                class_idx = np.argmax(tree.value[node][0])
                print("  " * depth + f"→ Predict: {self.class_names[class_idx]}")
                return
                
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            
            # Left branch (<=)
            print("  " * depth + f"If {feature} <= {threshold:.4f}:")
            print_tree_rules(tree, tree.children_left[node], depth + 1)
            
            # Right branch (>)
            print("  " * depth + f"If {feature} > {threshold:.4f}:")
            print_tree_rules(tree, tree.children_right[node], depth + 1)
        
        print_tree_rules(self.model.tree_)

    def evaluate_model(self, X_test=None, y_test=None):
        """
        Robust evaluation that handles all edge cases
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        y_pred = self.model.predict(X_test)
        
        # Get all possible classes from the label encoder
        all_classes = set(range(len(self.class_names)))
        present_classes = set(y_test) | set(y_pred)
        missing_classes = all_classes - present_classes
        
        # Warn about missing classes
        if missing_classes:
            missing_names = [self.class_names[i] for i in missing_classes]
            print(f"Warning: Missing classes in test data: {missing_names}")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # Only include present classes in classification report
        present_class_names = [self.class_names[i] for i in sorted(present_classes)]
        try:
            metrics['classification_report'] = classification_report(
                y_test, y_pred,
                labels=sorted(present_classes),
                target_names=present_class_names,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            print(f"Couldn't generate classification report: {str(e)}")
            metrics['classification_report'] = None
        
        # Calculate per-class F1 scores safely
        f1_scores = f1_score(
            y_test, y_pred, 
            labels=sorted(present_classes),
            average=None,
            zero_division=0
        )
        
        for i, class_idx in enumerate(sorted(present_classes)):
            metrics[f'f1_{self.class_names[class_idx]}'] = f1_scores[i]
        
        # Set missing classes to NaN
        for class_idx in missing_classes:
            metrics[f'f1_{self.class_names[class_idx]}'] = np.nan
            
        return metrics

    def plot_confusion_matrix(self, save_path=None, normalize=True):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        return None