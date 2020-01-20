from django import forms

from .models import FileUpload

# contiene formularios; con lo que se reciben campos desde el usuario

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ['file', 'date', 'ndays', 'nrooms',
                  'dia1AM', 'dia2AM', 'dia3AM', 'dia4AM', 'dia5AM',
                  'dia1PM', 'dia2PM', 'dia3PM', 'dia4PM', 'dia5PM']