from django.shortcuts import render, redirect
from django.views import View
from django.db.models import Sum, Avg, F, Count, Q
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.utils.translation import ugettext_lazy as _

# aca se importan las cosas desde los otros archivos .py (en la misma carpeta, con .)
from .forms import FileUploadForm
from .models import FileUpload, Ingreso, Schedule
from .utils import process_data, run_model, assign_list, assign_list2, assign_list3, normalizar_texto

import time
import math
import json
import datetime as dt

# exportar excel
import xlwt


# controla carga los templates para la vista de lo que esta en urls.py, y se conecta con los modelos

class IndexView(View):  # url vacia
    template = 'dashboard/index.html'  # template de pagina web
    context = {}  # lo que se va a enviar al template

    def get(self, request, *args, **kwargs):  # mostrar una pagina se hace con get; post para recibir input del usuario
        self.context['files'] = FileUpload.objects.order_by('-created')[
                                :5]  # trae todos los objetos del timpo FileUpload
        # y los ordena por campo created (una fecha)
        return render(request, self.template, self.context)  # manda a devolver la pagina con las cosas que se le ordena


class IndexViewAdvanced(View):  # url vacia
    context = {}  # lo que se va a enviar al template
    template = 'dashboard/avanzado.html'  # template de pagina web

    def get(self, request, *args, **kwargs):  # mostrar una pagina se hace con get; post para recibir input del usuario
        self.context['files'] = FileUpload.objects.order_by('-created')[
                                :5]  # trae todos los objetos del timpo FileUpload
        # y los ordena por campo created (una fecha)
        return render(request, self.template, self.context)  # manda a devolver la pagina con las cosas que se le ordena


class ScheduleView(View):
    context = {}
    template = 'dashboard/results.html'

    def get(self, request, id_result, *args, **kwargs):  # para mostrar info al usuari

        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            return render(request, self.template, self.context)

        self.context['schedule'] = Schedule.objects.filter(file=file)
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['especialidades'] = Schedule.objects.filter(file=file).values('especialidad'). \
            annotate(time=Sum('initial_duration')).annotate(time_h=F('time') / 60).order_by('-time')
        self.context['id_result'] = id_result

        return render(request, self.template, self.context)

    def post(self, request, *args, **kwargs):  # post es para recibir info del usuario

        form = FileUploadForm(request.POST, request.FILES)  # formulario, definidos en form.py.
        # Si hay un post, se usa un form
        if form.is_valid():

            if int(request.META['CONTENT_LENGTH']) > 10485760:  # si el tamaño es mayor a 10 megas
                self.template = 'dashboard/index.html'
                self.context['files'] = FileUpload.objects.order_by('-created')
                self.context['mensaje'] = "El archivo a subir debe ser menor a 10 MB"
                return render(request, self.template, self.context)

            programming_date = dt.datetime.strptime(request.POST['date'], "%d/%m/%Y").date()

            # request es el objetoi que contiene todas las cosas que vienen del request del usuario (cuando ejecuta la accion)
            # variables definidas en el html
            new_file = FileUpload(file=request.FILES['file'], date=programming_date, nrooms=request.POST['nrooms'],
                                  ndays=request.POST['ndays'],
                                  dia1AM=request.POST['dia1AM'],
                                  dia2AM=request.POST['dia2AM'],
                                  dia3AM=request.POST['dia3AM'],
                                  dia4AM=request.POST['dia4AM'],
                                  dia5AM=request.POST['dia5AM'],
                                  dia1PM=request.POST['dia1PM'],
                                  dia2PM=request.POST['dia2PM'],
                                  dia3PM=request.POST['dia3PM'],
                                  dia4PM=request.POST['dia4PM'],
                                  dia5PM=request.POST['dia5PM'],
                                  )

            start_time = time.time()
            Datos, missingColumns = process_data(new_file.file, programming_date)  # funcion definida en utils.py.

            if missingColumns:
                self.template = 'dashboard/index.html'
                self.context['files'] = FileUpload.objects.order_by('-created')
                self.context['mensaje'] = "Archivo sin las columnas " + ", ".join(
                    missingColumns)  # 'mensaje' tiene que estar en index.html
                return render(request, self.template, self.context)

            print(time.time() - start_time, 's')
            new_file.save()  # esto es lo que crea las filas en la base de datos de sqlite

            save_lista_espera(Datos, new_file)

            run_model(Datos, programming_date, new_file)  # del utils

            return redirect('/programacion/' + str(new_file.pk))  # redirige a la pagina de programacion
        else:
            self.template = 'dashboard/index.html'
            self.context['files'] = FileUpload.objects.order_by('-created')[:10]
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

        schedule = Schedule.objects.filter(file=file).annotate(
            utilization=(F('initial_duration') - F('remaining_duration'))
                        * 100 / F('initial_duration'))

        self.context['schedule'] = schedule
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['id_result'] = id_result

        ingreso = Ingreso.objects.filter(file=file)
        self.context['initial_mean'] = ingreso.aggregate(average=Avg('tiempoespera'))
        self.context['initial_median'] = median_value(ingreso, 'tiempoespera')
        ingreso_final = Ingreso.objects.filter(file=file).annotate(schedule_count=Count('schedule')).filter(
            schedule_count=0)
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

        schedule = Schedule.objects.filter(file=file).annotate(
            utilization=(F('initial_duration') - F('remaining_duration'))
                        * 100 / F('initial_duration'))

        self.context['schedule'] = schedule
        self.context['rooms'] = Schedule.objects.filter(file=file).values('room').distinct().order_by('room')
        self.context['days'] = Schedule.objects.filter(file=file).values('day').distinct()
        self.context['id_result'] = id_result

        ingreso = Ingreso.objects.filter(file=file)
        self.context['initial_mean'] = ingreso.aggregate(average=Avg('tiempoespera'))
        self.context['initial_median'] = median_value(ingreso, 'tiempoespera')
        ingreso_final = Ingreso.objects.filter(file=file).annotate(schedule_count=Count('schedule')).filter(
            schedule_count=0)
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

        schedule = Schedule.objects.filter(file=file,
                                           especialidad=especialidad)
        ingresos_duracion = Ingreso.objects.filter(file=file,
                                                   especialidad=especialidad,
                                                   prioridad=1).values('duracion') \
            .aggregate(time=Sum('duracion'), count=Count('duracion'))

        ingresos = Ingreso.objects.filter(file=file,
                                          especialidad=especialidad).order_by(
            '-tiempoespera')
        ingresos = ingresos.filter(~Q(prioridad=1)).exclude(duracion__isnull=True)

        tiempo_especialidad = 0
        for e in especialidades:
            if e['especialidad'] == especialidad:
                tiempo_especialidad = e['time']
                break

        ingresos_prioritarios = Ingreso.objects.filter(file=file,
                                                       especialidad=especialidad,
                                                       prioridad=1).order_by(
            '-tiempoespera')
        ingresos = Ingreso.objects.filter(file=file,
                                          especialidad=especialidad,
                                          prioridad=0).order_by(
            '-tiempoespera')

        tiempo_restante = assign_list3(file, schedule, ingresos, ingresos_prioritarios)

        # ingresos_duracion = Ingreso.objects.filter(file=file, especialidad=especialidad, prioridad=1).values('duracion') \
        #     .aggregate(time=Sum('duracion'), count=Count('duracion'))
        #
        # tiempo_restante = tiempo_especialidad - (ingresos_duracion['time'] + 15 * ingresos_duracion['count'])
        #        tiempo_restante = getmaximumtimeavailable(ingresos, ingresos_prioritarios, )

        self.context['tiempo_especialidad'] = tiempo_especialidad
        self.context['tiempo_restante'] = tiempo_restante
        self.context['especialidades'] = especialidades
        self.context['especialidad'] = especialidad
        self.context['ingresos'] = ingresos
        self.context['ingresos_prioritarios'] = ingresos_prioritarios
        self.context['id_result'] = id_result

        return render(request, self.template, self.context)


class PacientesViewFirstTime(View):
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

        if True:  # itera las lista prioritaria para tooodas las especialidades
            for e in especialidades:
                schedule = Schedule.objects.filter(file=file,
                                                   especialidad=e['especialidad'])
                ingresoss = Ingreso.objects.filter(file=file,
                                                   especialidad=e['especialidad']).order_by(
                    '-tiempoespera')
                ingresoss = ingresoss.filter(~Q(prioridad=1)).exclude(duracion__isnull=True)
                ingresoss_prioritarios = Ingreso.objects.filter(file=file,
                                                                especialidad=e['especialidad'],
                                                                prioridad=1).order_by(
                    '-tiempoespera')
                assign_list2(file, schedule, ingresoss, ingresoss_prioritarios)

        schedule = Schedule.objects.filter(file=file,
                                           especialidad=especialidad)
        ingresos_duracion = Ingreso.objects.filter(file=file,
                                                   especialidad=especialidad,
                                                   prioridad=1).values('duracion') \
            .aggregate(time=Sum('duracion'),
                       count=Count('duracion'))

        ingresos = Ingreso.objects.filter(file=file,
                                          especialidad=especialidad).order_by(
            '-tiempoespera')
        ingresos = ingresos.filter(~Q(prioridad=1)).exclude(duracion__isnull=True)

        tiempo_especialidad = 0
        for e in especialidades:
            if e['especialidad'] == especialidad:
                tiempo_especialidad = e['time']
                break

        ingresos_prioritarios = Ingreso.objects.filter(file=file,
                                                       especialidad=especialidad,
                                                       prioridad=1).order_by(
            '-tiempoespera')
        ingresos = Ingreso.objects.filter(file=file,
                                          especialidad=especialidad,
                                          prioridad=0).order_by(
            '-tiempoespera')

        tiempo_restante = assign_list3(file, schedule, ingresos, ingresos_prioritarios)

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
            schedule1 = Schedule.objects.get(file=file,
                                             room=room1,
                                             day=day1,
                                             bloque=bloque1)
            id_initial = schedule1.pk
            s1duration = schedule1.initial_duration
            schedule2 = Schedule.objects.filter(file=file,
                                                room=room2,
                                                day=day2,
                                                bloque=bloque2)

            if schedule2:
                s2duration = schedule2.first().initial_duration
                schedule2.update(room=room1,
                                 day=day1,
                                 bloque=bloque1,
                                 initial_duration=s1duration,
                                 remaining_duration=s1duration)

            if id_initial:
                Schedule.objects.filter(pk=id_initial).update(room=room2,
                                                              day=day2,
                                                              bloque=bloque2,
                                                              initial_duration=s2duration,
                                                              remaining_duration=s2duration)

            especialidades = list(Schedule.objects.filter(file=file).values('especialidad'). \
                annotate(time=Sum('initial_duration')).annotate(time_h=F('time') / 60).order_by(
                '-time'))

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
            tiempo_restante = data['tiempo_restante']
            idp = data['idp']
            mode = data['mode']
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))

        try:
            ingreso = Ingreso.objects.get(file=file, pk=int(idp))
            if mode == 'in':
                ingreso.prioridad = 1
            else:
                ingreso.prioridad = 0
            ingreso.save()

            schedule = Schedule.objects.filter(file=file,
                                               especialidad=especialidad)

            ingresos_prioritarios = Ingreso.objects.filter(file=file,
                                                           especialidad=especialidad,
                                                           prioridad=1).order_by(
                '-tiempoespera')
            ingresos = Ingreso.objects.filter(file=file,
                                              especialidad=especialidad,
                                              prioridad=0).order_by(
                '-tiempoespera')

            tiempo_restante = assign_list3(file, schedule, ingresos, ingresos_prioritarios)

        except:
            return HttpResponse(_('Not found or cannot update!'))

        return JsonResponse(tiempo_restante, safe=False)

    return HttpResponse(_('No post!'))


def tiempoCompleto(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            id_result = data['id_result']
            idp = data['idp']
            time = data['time']
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))

        try:
            boolean = '1'
            ingreso = Ingreso.objects.get(file=file,
                                          pk=int(idp))
            if (float(ingreso.duracion) > float(time)):
                boolean = '0'

        except:
            return HttpResponse(_('Not found or cannot update!'))

        return HttpResponse(_(boolean))

    return HttpResponse(_('No post!'))


@transaction.atomic  # operacion atomica: que se ejecute todo_ en el mismo momento
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
        return values[(count // 2)]
    else:
        return sum(values[count // 2 - 1:count // 2 + 1]) / 2


def export_xls(request, id_result):
    if request.method == 'GET':
        try:
            # id_result = request.GET.get('id_result', None)
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))
        try:
            response = HttpResponse(content_type='application/ms-excel')
            response['Content-Disposition'] = 'attachment; filename="Lista_de_Pacientes.xls"'
            especialidades = Schedule.objects.filter(file=file).values('especialidad').distinct()
            wb = xlwt.Workbook(encoding='utf-8')
            for e in especialidades:
                titulo = e['especialidad']
                ws = wb.add_sheet(titulo)
                row_num = 0
                font_style = xlwt.XFStyle()
                font_style.font.bold = True
                columns = ['RUN', 'OPERACION', 'TIEMPO_ESPERADO', 'DURACION',
                           '', '', 'RUN', 'OPERACION', 'TIEMPO_ESPERADO', 'DURACION', ]
                for col_num in range(len(columns)):
                    ws.write(row_num, col_num, columns[col_num], font_style)
                font_style = xlwt.XFStyle()
                rows_prior = Ingreso.objects.filter(file=file,
                                                    especialidad=e['especialidad'],
                                                    prioridad=1).values_list('run',
                                                                             'prestacion',
                                                                             'tiempoespera',
                                                                             'duracion').order_by('-tiempoespera')
                for row in rows_prior:
                    row_num += 1
                    for col_num in range(4):
                        ws.write(row_num, col_num, row[col_num], font_style)
                rows_no_prior = Ingreso.objects.filter(file=file,
                                                       especialidad=e['especialidad'],
                                                       prioridad=0).values_list('run',
                                                                                'prestacion',
                                                                                'tiempoespera',
                                                                                'duracion').order_by('-tiempoespera')
                row_num = 0
                for row in rows_no_prior:
                    row_num += 1
                    for col_num in range(4):
                        ws.write(row_num, col_num + 6, row[col_num], font_style)
            wb.save(response)

        except:
            return HttpResponse(_('Not found or cannot update!'))
        return response

    return HttpResponse(_('No post!'))


def export_xls2(request, id_result):
    if request.method == 'GET':
        try:
            file = FileUpload.objects.get(pk=id_result)
        except:
            return HttpResponse(_('Invalid request!'))
        try:
            response = HttpResponse(content_type='application/ms-excel')
            response['Content-Disposition'] = 'attachment; filename="Programación.xls"'

            dias = Schedule.objects.filter(file=file).values('day').distinct()
            salas = Schedule.objects.filter(file=file).values('room').distinct()
            wb = xlwt.Workbook(encoding='utf-8')
            font_style = xlwt.XFStyle()
            for d in dias:
                titulo = d['day']
                ws = wb.add_sheet(titulo)
                ncol = 1
                for s in salas:
                    nrow = 0
                    ws.write(nrow, ncol-1, '-', font_style)
                    ws.write(nrow, ncol, 'Sala '+str(s['room']), font_style)
                    ws.write(nrow, ncol+1, '-', font_style)
                    nrow += 2
                    especialidades_AM = Schedule.objects.filter(file=file,
                                                                day=d['day'],
                                                                bloque='AM',
                                                                room=s['room']).values('especialidad').distinct()
                    ws.write(nrow, ncol-1, '-', font_style)
                    ws.write(nrow, ncol, 'Bloque AM', font_style)
                    ws.write(nrow, ncol+1, '-', font_style)
                    nrow += 1
                    for e in especialidades_AM:
                        ws.write(nrow, ncol - 1, '-', font_style)
                        ws.write(nrow, ncol, e['especialidad'], font_style)
                        ws.write(nrow, ncol + 1, '-', font_style)
                        nrow += 1
                        schedule = Schedule.objects.filter(file=file,
                                                           especialidad=e['especialidad'],
                                                           day=d['day'],
                                                           bloque='AM',
                                                           room=s['room'])
                        for ss in schedule:
                            if ss.bloque_extendido == 0:
                                ws.write(nrow, ncol-1, 'Rut', font_style)
                                ws.write(nrow, ncol, 'Prestación', font_style)
                                ws.write(nrow, ncol+1, 'Duración (min)', font_style)
                                nrow += 1
                                rows_prior = Ingreso.objects.filter(file=file,
                                                                    especialidad=e['especialidad'],
                                                                    prioridad=1).order_by(
                                    '-tiempoespera')
                                for row in rows_prior:
                                    if ss in row.schedule.all():
                                        ws.write(nrow, ncol-1, row.run, font_style)
                                        ws.write(nrow, ncol, row.prestacion, font_style)
                                        ws.write(nrow, ncol+1, row.duracion, font_style)
                                        nrow += 1
                            elif ss.bloque_extendido == 1:
                                ws.write(nrow, ncol - 1, '-', font_style)
                                ws.write(nrow, ncol, 'BLOQUE EXTENDIDO', font_style)
                                ws.write(nrow, ncol + 1, '-', font_style)
                                nrow += 1
                                ws.write(nrow, ncol - 1, 'Rut', font_style)
                                ws.write(nrow, ncol, 'Prestación', font_style)
                                ws.write(nrow, ncol + 1, 'Duración (min)', font_style)
                                nrow += 1
                                rows_prior = Ingreso.objects.filter(file=file,
                                                                    especialidad=e['especialidad'],
                                                                    prioridad=1).order_by(
                                    '-tiempoespera')
                                for row in rows_prior:
                                    if ss in row.schedule.all():
                                        ws.write(nrow, ncol - 1, row.run, font_style)
                                        ws.write(nrow, ncol, row.prestacion, font_style)
                                        ws.write(nrow, ncol + 1, row.duracion, font_style)
                                        nrow += 1
                        nrow += 1
                    especialidades_PM = Schedule.objects.filter(file=file,
                                                                day=d['day'],
                                                                bloque='PM',
                                                                room=s['room']).values('especialidad').distinct()
                    ws.write(nrow, ncol-1, '-', font_style)
                    ws.write(nrow, ncol, 'Bloque PM', font_style)
                    ws.write(nrow, ncol+1, '-', font_style)
                    nrow += 1
                    for e in especialidades_PM:
                        ws.write(nrow, ncol - 1, '-', font_style)
                        ws.write(nrow, ncol, e['especialidad'], font_style)
                        ws.write(nrow, ncol + 1, '-', font_style)
                        nrow += 1
                        schedule = Schedule.objects.filter(file=file,
                                                           especialidad=e['especialidad'],
                                                           day=d['day'],
                                                           bloque='PM',
                                                           room=s['room'])
                        for ss in schedule:
                            if ss.bloque_extendido == 0:
                                ws.write(nrow, ncol - 1, 'Rut', font_style)
                                ws.write(nrow, ncol, 'Prestación', font_style)
                                ws.write(nrow, ncol + 1, 'Duración (min)', font_style)
                                nrow += 1
                                rows_prior = Ingreso.objects.filter(file=file,
                                                                    especialidad=e['especialidad'],
                                                                    prioridad=1).order_by(
                                    '-tiempoespera')
                                for row in rows_prior:
                                    if ss in row.schedule.all():
                                        ws.write(nrow, ncol - 1, row.run, font_style)
                                        ws.write(nrow, ncol, row.prestacion, font_style)
                                        ws.write(nrow, ncol + 1, row.duracion, font_style)
                                        nrow += 1

                            elif ss.bloque_extendido == 1:
                                ws.write(nrow, ncol - 1, '-', font_style)
                                ws.write(nrow, ncol, 'BLOQUE EXTENDIDO', font_style)
                                ws.write(nrow, ncol + 1, '-', font_style)
                                nrow += 1
                                ws.write(nrow, ncol - 1, '', font_style)
                                ws.write(nrow, ncol, '', font_style)
                                ws.write(nrow, ncol + 1, '', font_style)
                                nrow += 1
                        nrow += 1
                    ncol += 4
            wb.save(response)

        except:
            return HttpResponse(_('Not found or cannot update!'))
        return response
    return HttpResponse(_('No post!'))
