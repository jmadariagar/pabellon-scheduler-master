from django.urls import path

from .views import *

urlpatterns = [
    path('', IndexView.as_view(), name='index-pabellon'),
    path('programacion/<int:id_result>', ScheduleView.as_view(), name = 'programacion'),
    path('programacion', ScheduleView.as_view(), name='upload-file'),
    path('update-schedule', updateSchedule, name='update-schedule'),
    path('update-prioridad', updatePrioridad, name='update-prioridad'),
    path('programacion/<int:id_result>/lista', ListaView.as_view(), name = 'programacion-lista'),
    path('programacion/<int:id_result>/pacientes', PacientesView.as_view(), name = 'programacion-pacientes'),
    path('programacion/<int:id_result>/pacientes/<str:especialidad>', PacientesView.as_view(), name = 'programacion-pacientes-esp'),
]