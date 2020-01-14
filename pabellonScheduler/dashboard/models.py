from django.db import models

# Create your models here.

class FileUpload(models.Model): #el excel que se sube
    created = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='uploads/')
    date = models.DateField(null=True)
    ndays = models.IntegerField(null=True)
    ndaysAM = models.IntegerField(null=True)
    nrooms = models.IntegerField(null=True)
    hoursam = models.IntegerField(null=True)
    hourspm = models.IntegerField(null=True)
    alfa_at_cerr = models.IntegerField(null=True, default=10000)
    alfa_ges = models.IntegerField(null=True, default=1000)
    alfa_reprog = models.IntegerField(null=True, default=100)
    alfa_clinic_prior = models.IntegerField(null=True, default=10)
    alfa_tiempo_espera = models.IntegerField(null=True, default=1)

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
    orden = models.FloatField(null=True, default=0)
    schedule = models.ManyToManyField(Schedule)
    file = models.ForeignKey(FileUpload, on_delete=models.CASCADE)

