from django import forms
from .models import DataSet

class ProcessingForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = ['dimension','features','classifier']

       # widgets = {
        #    'dimension': forms.ChoiceField(widget=forms.RadioSelect),
        #    'features': forms.ChoiceField(widget=forms.RadioSelect),
        #    'classifier': forms.ChoiceField(widget=forms.RadioSelect),
        #}
