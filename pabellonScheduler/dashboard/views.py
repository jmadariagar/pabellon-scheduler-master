from django.shortcuts import render, redirect
from django.views import View
from django.db.models import Sum, Avg, F, Count, Q
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.utils.translation import ugettext_lazy as _

from .forms import FileUploadForm
from .models import FileUpload, Ingreso, Schedule
from .utils import process_data, run_model, assign_list

import time
import math
import json
import datetime as dt

class IndexView(View):
    template = 'dashboard/index.html'
    context = {}

    def get(self, request, *args, **kwargs):
        self.context['files'] = FileUpload.objects.order_by('-created')[:10]
        return render(request, self.template, self.context)


class ScheduleView(View):

    context = {}
    template = 'dashboard/results.html'

    def get(self, request, id_result, *args, **kwargs):

        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            return render(request, self.template, self.context)

        self.context['schedule'] = Schedule.objects.filter(file=file)
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['especialidades'] = Schedule.objects.filter(file=file).values('especialidad').\
            annotate(time=Sum('initial_duration')).annotate(time_h=F('time')/60).order_by('-time')
        self.context['id_result'] = id_result

        return render(request, self.template, self.context)

    def post(self, request, *args, **kwargs):

        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():

            if int(request.META['CONTENT_LENGTH']) > 10485760:
                self.template = 'dashboard/index.html'
                self.context['files'] = FileUpload.objects.order_by('-created')[:10]
                self.context['mensaje'] = "El archivo a subir debe ser menor a 10 MB"
                return render(request, self.template, self.context)

            programming_date = dt.datetime.strptime(request.POST['date'], "%d/%m/%Y").date()

            new_file = FileUpload(file=request.FILES['file'], date=programming_date, nrooms=request.POST['nrooms'],
                                  ndays=request.POST['ndays'], hoursam=request.POST['hoursam'],
                                  hourspm=request.POST['hourspm'])

            start_time = time.time()
            Datos, missingColumns = process_data(new_file.file, programming_date)

            if missingColumns:
                self.template = 'dashboard/index.html'
                self.context['files'] = FileUpload.objects.order_by('-created')
                self.context['mensaje'] = "Archivo sin las columnas " + ", ".join(missingColumns)
                return render(request, self.template, self.context)

            print(time.time() - start_time, 's')
            new_file.save()

            save_lista_espera(Datos, new_file)

            run_model(Datos, programming_date, new_file)

            return redirect('/programacion/' + str(new_file.pk))
        else:
            self.template = 'dashboard/index.html'
            self.context['files'] = FileUpload.objects.order_by('-created')
            self.context['mensaje'] = "Error al subir el archivo"
            return render(request, self.template, self.context)

class ListaView(View):

    context = {}
    template = 'dashboard/lista.html'

    def get(self, request, id_result, *args, **kwargs):
        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            template = 'dashboard/index.html'
            return render(request, template, self.context)

        schedule = Schedule.objects.filter(file=file).annotate(utilization=(F('initial_duration') - F('remaining_duration'))
                        * 100 / F('initial_duration'))

        self.context['schedule'] = schedule
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['id_result'] = id_result

        ingreso = Ingreso.objects.filter(file=file)
        self.context['initial_mean'] = ingreso.aggregate(average=Avg('tiempoespera'))
        self.context['initial_median'] = median_value(ingreso, 'tiempoespera')
        ingreso_final = Ingreso.objects.filter(file=file).annotate(schedule_count=Count('schedule')).filter(schedule_count=0)
        self.context['final_mean'] = ingreso_final.aggregate(average=Avg('tiempoespera'))
        self.context['final_median'] = median_value(ingreso_final, 'tiempoespera')

        return render(request, self.template, self.context)

    def post(self, request, id_result, *args, **kwargs):

        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            template = 'dashboard/index.html'
            return render(request, template, self.context)

        schedule = Schedule.objects.filter(file=file)
        ingresos = Ingreso.objects.filter(file=file, prioridad=0).order_by('-tiempoespera')
        ingresos_prioritarios = Ingreso.objects.filter(file=file, prioridad=1).order_by('-tiempoespera')

        assign_list(file, schedule, ingresos, ingresos_prioritarios)

        schedule = Schedule.objects.filter(file=file).annotate(utilization=(F('initial_duration') - F('remaining_duration'))
                        * 100 / F('initial_duration'))

        self.context['schedule'] = schedule
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['id_result'] = id_result

        ingreso = Ingreso.objects.filter(file=file)
        self.context['initial_mean'] = ingreso.aggregate(average=Avg('tiempoespera'))
        self.context['initial_median'] = median_value(ingreso, 'tiempoespera')
        ingreso_final = Ingreso.objects.filter(file=file).annotate(schedule_count=Count('schedule')).filter(schedule_count=0)
        self.context['final_mean'] = ingreso_final.aggregate(average=Avg('tiempoespera'))
        self.context['final_median'] = median_value(ingreso_final, 'tiempoespera')

        return render(request, self.template, self.context)

class PacientesView(View):

    context = {}
    template = 'dashboard/pacientes.html'

    def get(self, request, id_result, *args, **kwargs):

        # checkear si viene chat id en la url
        especialidad = None
        if 'especialidad' in kwargs:
            especialidad = kwargs['especialidad']

        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            template = 'dashboard/index.html'
            return render(request, template, self.context)

        especialidades = Schedule.objects.filter(file=file).values('especialidad'). \
                              annotate(time=Sum('initial_duration')).order_by('-time')

        if not especialidad:
            especialidad = especialidades.first()['especialidad']

        ingresos_duracion = Ingreso.objects.filter(file=file, especialidad=especialidad, prioridad=1).values('duracion') \
            .aggregate(time=Sum('duracion'), count=Count('duracion'))

        ingresos = Ingreso.objects.filter(file=file, especialidad=especialidad).order_by('-tiempoespera')
        ingresos = ingresos.filter(~Q(prioridad=1)).exclude(duracion__isnull=True)

        tiempo_especialidad = 0
        for e in especialidades:
            if e['especialidad'] == especialidad:
                tiempo_especialidad = e['time']
                break

        if ingresos_duracion['time'] and ingresos_duracion['count']:
            tiempo_restante = tiempo_especialidad - (ingresos_duracion['time'] + 15*ingresos_duracion['count'])
        else:
            tiempo_restante = tiempo_especialidad

        if tiempo_restante > 0:
            for i in ingresos:
                if i.duracion and tiempo_restante > i.duracion + 15:
                    i.prioridad = 1
                    i.save()
                    tiempo_restante = tiempo_restante - (i.duracion + 15)

        ingresos_prioritarios = Ingreso.objects.filter(file=file, especialidad=especialidad, prioridad=1).order_by(
            '-tiempoespera')
        ingresos = Ingreso.objects.filter(file=file, especialidad=especialidad, prioridad=0).order_by(
            '-tiempoespera')

        self.context['tiempo_especialidad'] = tiempo_especialidad
        self.context['tiempo_restante'] = tiempo_restante
        self.context['especialidades'] = especialidades
        self.context['especialidad'] = especialidad
        self.context['ingresos'] = ingresos
        self.context['ingresos_prioritarios'] = ingresos_prioritarios
        self.context['id_result'] = id_result

        return render(request, self.template, self.context)


def updateSchedule(request):

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            id_result = data['id_result']
            room1 = data['room1']
            room2 = data['room2']
            day1 = data['day1']
            day2 = data['day2']
            bloque1 = data['bloque1']
            bloque2 = data['bloque2']
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))

        try:
            schedule1 = Schedule.objects.get(file=file, room=room1, day=day1, bloque=bloque1)
            id_initial = schedule1.pk
            s1duration = schedule1.initial_duration
            schedule2 = Schedule.objects.filter(file=file, room=room2, day=day2, bloque=bloque2)

            if schedule2:
                s2duration = schedule2.first().initial_duration
                schedule2.update(room=room1, day=day1, bloque=bloque1, initial_duration=s1duration,
                                 remaining_duration=s1duration)
            if id_initial:
                Schedule.objects.filter(pk = id_initial).update(room=room2, day=day2, bloque=bloque2,
                                                                initial_duration=s2duration, remaining_duration=s2duration)

            especialidades = list(Schedule.objects.filter(file=file).values('especialidad'). \
                annotate(time=Sum('initial_duration')).annotate(time_h=F('time') / 60).order_by('-time'))

        except:
            return HttpResponse(_('Not found or cannot update!'))

        return JsonResponse(especialidades, safe=False)

    return HttpResponse(_('No post!'))

def updatePrioridad(request):

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            id_result = data['id_result']
            especialidad = data['especialidad']
            tiempo_especialidad = data['tiempo_especialidad']
            idp = data['idp']
            mode = data['mode']
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))

        try:
            ingreso = Ingreso.objects.get(file=file, pk=int(idp))
            print(ingreso)
            if mode == 'in':
                print('in')
                ingreso.prioridad = 1
            else:
                print('out')
                ingreso.prioridad = 0
            ingreso.save()

            ingresos_duracion = Ingreso.objects.filter(file=file, especialidad=especialidad, prioridad=1).values('duracion')\
                .aggregate(time=Sum('duracion'), count=Count('duracion'))

            if ingresos_duracion['time'] and ingresos_duracion['count']:
                tiempo_restante = tiempo_especialidad - (ingresos_duracion['time'] + 15 * ingresos_duracion['count'])
            else:
                tiempo_restante = tiempo_especialidad

        except:
            return HttpResponse(_('Not found or cannot update!'))

        return JsonResponse(tiempo_restante, safe=False)

    return HttpResponse(_('No post!'))

@transaction.atomic
def save_lista_espera(content, file):
    for index, item in content.iterrows():

        for key, it in enumerate(item):
            if (isinstance(it, int) or isinstance(it, float)) and math.isnan(it):
                item[key] = None

        operacion = Ingreso(run=item['RUN'],
                            id_intervencion=item['ID'],
                            prestacion=item['PRESTA_MIN'],
                            especialidad=item['PRESTA_EST'],
                            fechaingreso=item['F_ENTRADA'],
                            tiempoespera=item['Waiting_Time'],
                            duracion=item['MAIN_DURATION'],
                            file=file)
        operacion.save()

def median_value(queryset, term):
    count = queryset.count()
    values = queryset.values_list(term, flat=True).order_by(term)
    if count % 2 == 1:
        return values[(count//2)]
    else:
        return sum(values[count//2-1:count//2+1])/2
