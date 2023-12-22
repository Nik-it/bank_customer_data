# your_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict_and_generate_offers/', views.predict_and_generate_offers, name='predict_and_generate_offers'),
    # Add other URL patterns as needed
]
