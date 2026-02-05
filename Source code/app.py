import gradio as gr
import pickle
import pandas as pd
import numpy as np

def load_model():
    """Load the trained XGBoost model and metadata"""
    try:
        # Try different possible model file names
        model_files = [
            "criteo_attribution_xgboost_model.pkl",
            "XGBOOST/criteo_attribution_xgboost_model.pkl", 
            "xgboost_trained_model.pkl",
            "XGBOOST/xgboost_trained_model.pkl"
        ]
        
        for model_file in model_files:
            try:
                with open(model_file, "rb") as f:
                    saved_data = pickle.load(f)
                
                if isinstance(saved_data, dict):
                    model = saved_data.get('model')
                    feature_names = saved_data.get('feature_names', [])
                    metadata = saved_data.get('metadata', {})
                else:
                    model = saved_data
                    feature_names = []
                    metadata = {}
                
                print(f"Model loaded from {model_file}")
                return model, feature_names, metadata
            except:
                continue
        
        print("No model file found")
        return None, [], {}
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, [], {}

# Load model
model, feature_names, metadata = load_model()

def predict_attribution(click, campaign_perf, cost, cpo):
    """
    Predict attribution probability using realistic parameter ranges from actual Criteo data
    
    Based on analysis of 442,424 actual attributed cases:
    - Cost range: 0.000010 to 0.057756 (median: 0.000366)
    - CPO range: 0.004000 to 0.978832 (median: 0.078163) 
    - Campaign Performance: 0.001100 to 0.190200 (median: 0.043300)
    - Click: Must be 1 for attribution (100% of attributed cases had click=1)
    """
    
    if model is None:
        return "Error: Model not loaded", "N/A", "N/A", 0.0, "Model loading failed"
    
    # Create input with realistic feature mapping
    input_data = {
        "click": int(click),
        "campaign_attribution_rate": float(campaign_perf),  # This maps to campaign performance
        "cost": float(cost),
        "cpo": float(cpo)
    }
    
    # Add all other features as 0 (default values)
    if feature_names:
        for feat in feature_names:
            if feat not in input_data:
                input_data[feat] = 0
        input_df = pd.DataFrame([input_data])[feature_names]
    else:
        input_df = pd.DataFrame([input_data])
    
    try:
        # Get prediction probability
        prob_array = model.predict_proba(input_df)
        prob = float(prob_array[0][1])  # Probability of attribution (class 1)
        
        # Get binary prediction
        prediction = int(model.predict(input_df)[0])
        
        # Format outputs
        prob_pct = f"{prob:.2%}"
        pred_text = "ATTRIBUTED" if prediction == 1 else "NOT ATTRIBUTED"
        confidence = float(max(prob, 1-prob))
        conf_pct = f"{confidence:.2%}"
        
        # Business interpretation based on actual data insights
        if prob >= 0.5:
            interpretation = f"HIGH likelihood ({prob:.1%}) - Strong attribution signal detected"
        elif prob >= 0.05:  # Above average attribution rate of 2.69%
            interpretation = f"ABOVE AVERAGE likelihood ({prob:.1%}) - Better than typical 2.69% baseline"
        else:
            interpretation = f"LOW likelihood ({prob:.1%}) - Attribution unlikely"
        
        return prob_pct, pred_text, conf_pct, prob, interpretation
        
    except Exception as e:
        return "Error", str(e), "N/A", 0.0, "Prediction failed"

# Model performance info
model_info_text = "Model: XGBoost Classifier"
if metadata:
    perf = metadata.get('performance', {})
    roc_auc = perf.get('roc_auc', 0)
    precision = perf.get('precision', 0)
    recall = perf.get('recall', 0)
    
    if roc_auc > 0:
        model_info_text = f"ROC-AUC: {roc_auc:.3f} | Precision: {precision:.1%} | Recall: {recall:.1%}"

# Custom CSS matching your old style
css = """
.header {text-align: center; padding: 20px;}
footer {display: none !important;}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", secondary_hue="purple"), css=css) as demo:
    
    # Header (matching your old style)
    gr.HTML("""
    <div class="header">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 3em; margin: 0;">
            Criteo Attribution Prediction
        </h1>
        <p style="font-size: 1.2em; color: #666; margin-top: 10px;">
            AI-Powered Digital Advertising Attribution Analysis
        </p>
    </div>
    """)
    
    gr.Markdown(f"**Model Performance:** {model_info_text}")
    gr.Markdown("**Dataset:** 16.4M Criteo ad impressions | **CDAC PG-DBDA Final Project**")
    
    with gr.Tabs():
        with gr.Tab("Predict"):
            
            gr.Markdown("""
            ### Based on Analysis of 442,424 Real Attribution Cases
            **Key Finding:** 100% of attributed cases had click=1 (no attribution without clicks)  
            **Typical Ranges:** Cost: 0.000125-0.001021 | CPO: 0.011-0.164 | Campaign Perf: 2.2%-11.6%
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### User Engagement")
                    gr.Markdown("*100% of attributed cases had click=1*")
                    click = gr.Radio(
                        choices=[0, 1], 
                        value=1, 
                        label="Click Occurred",
                        info="Did user click?"
                    )
                    
                    gr.Markdown("### Campaign Metrics")
                    gr.Markdown("*Real range: 0.11% to 19.02% (median: 4.33%)*")
                    campaign_perf = gr.Slider(
                        0.001, 0.20, 0.043, 
                        step=0.001,
                        label="Campaign Attribution Rate"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Cost Data")
                    gr.Markdown("*Real range: 0.00001 to 0.058 (median: 0.000366)*")
                    cost = gr.Number(
                        value=0.000366, 
                        label="Ad Cost"
                    )
                    
                    gr.Markdown("*Real range: 0.004 to 0.979 (median: 0.078)*")
                    cpo = gr.Number(
                        value=0.078, 
                        label="Cost Per Order"
                    )
                    
                    predict_btn = gr.Button(
                        "Predict Attribution", 
                        variant="primary", 
                        size="lg", 
                        scale=2
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                prob_output = gr.Textbox(label="Attribution Probability", interactive=False, scale=1)
                pred_output = gr.Textbox(label="Prediction", interactive=False, scale=1)
                conf_output = gr.Textbox(label="Confidence", interactive=False, scale=1)
            
            prob_bar = gr.Slider(0.0, 1.0, 0.0, label="Probability", interactive=False, show_label=False)
            interpretation = gr.Textbox(label="Interpretation", interactive=False, lines=2)
            
            gr.Markdown("""
            ### Try These Real Data Examples:
            - **Typical Low:** Click=1, Campaign=2.2%, Cost=0.000125, CPO=0.011
            - **Median:** Click=1, Campaign=4.3%, Cost=0.000366, CPO=0.078
            - **High:** Click=1, Campaign=11.6%, Cost=0.001021, CPO=0.164
            - **No Click:** Click=0, Campaign=5%, Cost=0.001, CPO=0.1
            """)
        
        with gr.Tab("Model Info"):
            gr.Markdown("""
            ### XGBoost Attribution Model
            ( **Key Features:**
            - Click behavior: 96.9% feature importance
            - Campaign performance metrics
            - Contextual category analysis
            - Cost and efficiency data
            
            **Training Details:**
            - Dataset: 16.4 million impressions
            - Attribution rate: 2.69%
            - Class imbalance: 36:1
            - Attribution cases analyzed: 442,424
            
            **Real Data Ranges (from 442,424 attributed cases):**
            - **Click**: 100% of attributed cases had click=1
            - **Campaign Performance**: 0.11% - 19.02% (median: 4.33%)
            - **Cost**: 0.00001 - 0.058 (median: 0.000366)
            - **CPO**: 0.004 - 0.979 (median: 0.078)
            
            **Feature Engineering:**
            - Production-ready features
            - Real-time bidding compatible
            - No user-specific dependencies
            
            **Model Selection:**
            - Algorithm: XGBoost Classifier
            - Optimized for imbalanced data
            - Selected over: LightGBM, Random Forest, CatBoost, Logistic Regression
            
            **Business Applications:**
            - **High Probability (>5%)**: Above average - increase bids
            - **Medium Probability (1-5%)**: Around average - standard bidding
            - **Low Probability (<1%)**: Below average - reduce bids
            
            ### CDAC PG-DBDA Final Project
            **Project:** Criteo Attribution Modeling for Bidding  
            **Domain:** Digital Advertising & Programmatic Bidding  
            **Date:** January 2026
            """)
        
        # ←——— Added Tableau Dashboard Tab
        with gr.Tab("Tableau Dashboard"):
            gr.Markdown("""
            ## Tableau Attribution Dashboard

            Click the link below to open your interactive Tableau visualization:
            """)

            gr.Markdown(
                "[Open Tableau Dashboard](https://public.tableau.com/views/Criteo_Visualization_17701408628490/CampaignPerformanceAttributionOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)"
            )

    # Connect prediction function
    predict_btn.click(
        fn=predict_attribution,
        inputs=[click, campaign_perf, cost, cpo],
        outputs=[prob_output, pred_output, conf_output, prob_bar, interpretation]
    )

if __name__ == "__main__":
    demo.launch()
