from django import forms
from .models import DataSet

class ProcessingForm(forms.ModelForm):
    class Meta:

        model = DataSet
        fields = ['dimension','features','classifier']

        widgets = {
            'dimension': forms.RadioSelect(attrs={'id':'value'}),
            'features': forms.RadioSelect(attrs={'id':'value'}),
            'classifier': forms.RadioSelect(attrs={'id':'value'}),
        }


