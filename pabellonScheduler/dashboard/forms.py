from django import forms

from .models import FileUpload

# contiene formularios; con lo que se reciben campos desde el usuario

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ['file', 'date', 'ndays', 'nrooms', 'hoursam', 'hourspm']