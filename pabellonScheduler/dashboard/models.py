from django.db import models

# Create your models here.

class FileUpload(models.Model): #el excel que se sube
    created = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='uploads/')
    date = models.DateField(null=True)
    ndays = models.IntegerField(null=True, default=5)
    nrooms = models.IntegerField(null=True)
    dia1AM = models.IntegerField(null=True, default=5)
    dia2AM = models.IntegerField(null=True, default=5)
    dia3AM = models.IntegerField(null=True, default=5)
    dia4AM = models.IntegerField(null=True, default=5)
    dia5AM = models.IntegerField(null=True, default=5)
    dia1PM = models.IntegerField(null=True, default=3)
    dia2PM = models.IntegerField(null=True, default=3)
    dia3PM = models.IntegerField(null=True, default=3)
    dia4PM = models.IntegerField(null=True, default=3)
    dia5PM = models.IntegerField(null=True, default=2)

class Schedule(models.Model): #bloque
    especialidad = models.CharField(null=True, max_length=100)
    day = models.CharField(null=True, max_length=15)
    bloque = models.CharField(null=True, max_length=15)
    room = models.IntegerField(null=True)
    initial_duration = models.IntegerField(null=True)
    remaining_duration = models.IntegerField(null=True)
    bloque_extendido = models.IntegerField(null=True, default=0)
    file = models.ForeignKey(FileUpload, on_delete=models.CASCADE)

class Ingreso(models.Model): # cada entrada/persona que espera intervencion, del archivo que se sube
    run = models.CharField(null=True, max_length=15)
    id_intervencion = models.CharField(null=True, max_length=100)
    prestacion = models.CharField(null=True, max_length=15)
    especialidad = models.CharField(null=True, max_length=100)
    fechaingreso = models.DateField(null=True)
    tiempoespera = models.IntegerField(null=True)
    duracion = models.IntegerField(null=True)
    prioridad = models.IntegerField(null=True, default=0)
    schedule = models.ManyToManyField(Schedule)
    file = models.ForeignKey(FileUpload, on_delete=models.CASCADE)

