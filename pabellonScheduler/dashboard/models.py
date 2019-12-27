from django.db import models

# Create your models here.

class FileUpload(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='uploads/')
    date = models.DateField(null=True)
    ndays = models.IntegerField(null=True)
    nrooms = models.IntegerField(null=True)
    hoursam = models.IntegerField(null=True)
    hourspm = models.IntegerField(null=True)

class Schedule(models.Model):
    especialidad = models.CharField(null=True, max_length=100)
    day = models.CharField(null=True, max_length=15)
    bloque = models.CharField(null=True, max_length=15)
    room = models.IntegerField(null=True)
    initial_duration = models.IntegerField(null=True)
    remaining_duration = models.IntegerField(null=True)
    file = models.ForeignKey(FileUpload, on_delete=models.CASCADE)

class Ingreso(models.Model):
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

