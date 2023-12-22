# your_app/views.py
from django.shortcuts import render
import pandas as pd
import joblib
from .your_ml_module import make_predictions, preprocess_input, generate_personalized_offers

training_data = pd.read_csv('/home/nikit/djangochat/bank_customer_data.csv')
X_train = training_data.drop('Services_Taken', axis=1)

def predict_and_generate_offers(request):
    if request.method == 'POST':
        # Assuming you have a form with input fields in your_template.html
        user_id = int(request.POST.get('user_id', 0))  # Defaulting to 0 if the value is not provided or not an integer
        recent_activities = int(request.POST.get('recent_activities'))
        purchase_history = int(request.POST.get('purchase_history'))
        account_type = request.POST.get('account_type')
        money_on_account = float(request.POST.get('money_on_account'))
        age = int(request.POST.get('age'))
        monthly_income = float(request.POST.get('monthly_income'))
        credit_score = int(request.POST.get('credit_score'))
        number_of_dependents = int(request.POST.get('number_of_dependents'))
        education_level = request.POST.get('education_level')
        gender = request.POST.get('gender')

        # Prepare input features for prediction
        input_features = {
            'User_ID': user_id,
            'Recent_Activities': recent_activities,
            'Purchase_History': purchase_history,
            'Account_Type': account_type,
            'Money_on_Account': money_on_account,
            'Age': age,
            'Monthly_Income': monthly_income,
            'Credit_Score': credit_score,
            'Number_of_Dependents': number_of_dependents,
            'Education_Level': education_level,
            'Gender': gender,
        }

        # Preprocess the input features
        input_df = pd.DataFrame([input_features])
        processed_input = preprocess_input(input_df, X_train.columns)

        # Load the trained machine learning model
        pipeline = joblib.load('/home/nikit/djangochat/your_trained_model.pkl')

        # Assuming your RandomForestClassifier step is named 'model' in the pipeline
        random_forest_classifier = pipeline.named_steps['model']

        # Obtain the correct column order from processed_input
        column_order = processed_input.columns

        # Make predictions using the model
        predictions = make_predictions(random_forest_classifier, processed_input, column_order, top_n=10)

        # Generate personalized offers based on the predictions
        personalized_offers = generate_personalized_offers(predictions)

        # Pass the personalized offers to the template
        context = {'personalized_offers': personalized_offers}
        return render(request, 'myapp/your_template.html', context)

    # Render the initial form if it's a GET request
    return render(request, 'myapp/your_template.html')
