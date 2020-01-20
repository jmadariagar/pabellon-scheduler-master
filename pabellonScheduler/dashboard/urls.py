from django.urls import path

from .views import *

# como las url redireccionan a cosas que estan en views.py


urlpatterns = [
    path('', IndexView.as_view(), name='index-pabellon'), # url vacia llega aca
    path('advanced', IndexViewAdvanced.as_view(), name='index-avanzado'), # url vacia llega aca
    path('programacion/<int:id_result>', ScheduleView.as_view(), name = 'programacion'),
    path('programacion', ScheduleView.as_view(), name='upload-file'),
    path('programacion/avanzado', ScheduleView.as_view(), name='avanzado'),
    path('update-schedule', updateSchedule, name='update-schedule'),
    path('update-prioridad', updatePrioridad, name='update-prioridad'),
    path('tiempo-completo', tiempoCompleto, name='tiempo-completo'),
    path('programacion/<int:id_result>/lista', ListaView.as_view(), name = 'programacion-lista'),
    path('programacion/<int:id_result>/pacientes', PacientesViewFirstTime.as_view(), name='programacion-pacientes'),
    path('programacion/<int:id_result>/pacientes', PacientesView.as_view(), name = 'programacion-pacientes'),
    path('programacion/<int:id_result>/pacientes/<str:especialidad>', PacientesView.as_view(), name = 'programacion-pacientes-esp'),
    path('export_xls', export_xls, name='export_xls'),
    path('export_xls/<int:id_result>', export_xls, name='export_xls'),
    path('export_xls2/<int:id_result>', export_xls2, name='export_xls2'),
]