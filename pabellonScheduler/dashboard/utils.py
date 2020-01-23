import pandas as pd
import numpy as np
from datetime import timedelta
import time
import re
from unicodedata import normalize
import pickle
import pulp as plp
import math
from .models import Schedule
from itertools import cycle


# aca va todas las cosas que hicimos en los notebook traducidas a .py
# funciones que se usen en views.py

def process_data(file, programming_date):

    # Leo excel
    Datos = pd.read_excel(file)

    columns = list(Datos.columns.values)

    for c in columns:
        Datos.rename(columns={c: re.sub('[^a-zA-Z_]+', '', c).upper()}, inplace=True)


    columnsUploaded =  list(Datos.columns.values)
    columnsNeeded = ['RUN', 'PRESTA_MIN', 'PRESTA_EST', 'F_ENTRADA']

    missingColumns = [c for c in columnsNeeded if c not in columnsUploaded]

    if missingColumns:
        return None, missingColumns

    Datos = Datos[columnsNeeded]

    # parse variables
    Datos['F_ENTRADA'] = pd.to_datetime(Datos['F_ENTRADA'], format='%d-%m-%Y').dt.date
    Datos['RUN'] = Datos['RUN'].astype(str)
    Datos['RUN'] = Datos['RUN'].apply(digito_verificador)
    Datos['PRESTA_EST'] = Datos['PRESTA_EST'].apply(normalizar_texto)
    Datos = Datos.dropna()

    ## crear identidad
    for row in Datos.index:

        code = Datos.at[row, 'PRESTA_MIN']
        if type(code) == str:
            a, b, c = code.split('-')
            code = int(a + b + c)
            Datos.at[row, 'PRESTA_MIN'] = str(code) #la nueva base de datos para calcular los tiempos esta en str
                                                    #la antigua esta en int ()
        else:
            Datos.at[row, 'PRESTA_MIN'] = '0'

        yearE = Datos.at[row, 'F_ENTRADA'].year
        monthE = Datos.at[row, 'F_ENTRADA'].month
        dayE = Datos.at[row, 'F_ENTRADA'].day

        identity = str(row) + str(Datos.at[row, 'RUN']) + str(yearE) + str(monthE) + str(dayE)
        Datos.at[row, 'ID'] = identity

        waiting_time = (programming_date - Datos.at[row, 'F_ENTRADA']).days
        Datos.at[row, 'Waiting_Time'] = waiting_time

    Datos.reset_index(drop=True, inplace=True)
    Datos.drop_duplicates(['ID'], inplace=True)

    file_name = 'parameters/Extracted Parameters_new'
    with open(file_name, 'rb') as file_object:
        parameters = pickle.load(file_object)

    parameters = parameters[['MAIN_DURATION']]
    Datos = Datos.merge(parameters, how='left', left_on='PRESTA_MIN', right_index=True)

    return Datos, missingColumns


def run_model(queue, programming_date, file):  # Este es el que corre para

    file_name = 'parameters/Extracted Parameters_new'
    with open(file_name, 'rb') as file_object:
        parameters = pickle.load(file_object)

    print(parameters.columns)

    queue.drop(labels=['RUN'], axis=1, inplace=True)
    queue.set_index(keys='ID', inplace=True)

    queue.rename(index=str, columns={'PRESTA_MIN': 'Operation', 'PRESTA_EST': 'Service', 'F_ENTRADA': 'Arrival_Date'
                                     }, inplace=True)

    parameters.fillna(0, inplace=True)

    freq = queue['Waiting_Time']
    print('Size of the queue: ' + str(len(freq)))
    print('Average waiting time: ' + str(np.mean(freq)))
    print('Median waiting time: ' + str(np.median(freq)))

    specialties = queue['Service'].apply(normalizar_texto).unique()
    S = len(specialties)

    operations = parameters.index

    T = int(file.ndays)
    R = int(file.nrooms)


    weekdays = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    starting_date = programming_date
    starting_day = weekdays[starting_date.weekday()]

    estimator = 'MAIN_DURATION'  # Estimation according to the main operation. Yet the only relevant to use.

    times = range(1, T + 1)     # indice correspondiente a los dias
    rooms = range(1, R + 1)     # indice correspondiente a las salas

    queues = dict()
    for s in specialties:
        queues[s] = queue[queue['Service'] == s]


    def duration(patient):
        if type(patient) == int:
            operation = queue.at[patient, 'Operation']
            duration = parameters.at[operation, estimator]

        # This option is if one use the patient's ID as the argument
        else:
            operation = queue.at[patient, 'Operation']
            if operation in parameters.index:
                duration = parameters.at[operation, estimator]
            else:
                duration = 0

        return duration


    def waitingTime(patient):
        return queue.at[patient, 'Waiting_Time']


    def isService(patient, specialty):
        if queue.at[patient, 'Service'] == specialty:
            return 1
        else:
            return 0

    def weekday(t):
        if starting_day == 'Lunes':
            shift = 0
        elif starting_day == 'Martes':
            shift = 1
        elif starting_day == 'Miércoles':
            shift = 2
        elif starting_day == 'Jueves':
            shift = 3
        else:
            shift = 4

        return weekdays[(shift + t - 1) % 5]


    # Friday does not have the same time capacity as other days
    def isFriday(t):
        if weekday(t) == 'Viernes':
            return 1
        else:
            return 0

    u = 15              # tiempo de limpieza

    # Regular day

    def q_AM(t):
        if t == 1:
            return int(file.dia1AM)
        elif t == 2:
            return int(file.dia2AM)
        elif t == 3:
            return int(file.dia3AM)
        elif t == 4:
            return int(file.dia4AM)
        elif t == 5:
            return int(file.dia5AM)

    def q_PM(t):
        if t == 1:
            return int(file.dia1PM)
        elif t == 2:
            return int(file.dia2PM)
        elif t == 3:
            return int(file.dia3PM)
        elif t == 4:
            return int(file.dia4PM)
        elif t == 5:
            return int(file.dia5PM)


    requiredTime = {}

    for specialty in specialties:
        subtable = queue[queue['Service'] == specialty]         # escoje la tabla donde el servicio corresponde a la especialidad
        subtable['Duration'] = subtable.index
        subtable['Duration'] = subtable['Duration'].apply(duration)
        requiredTime[specialty] = np.sum(subtable['Duration'])  # tiempo requerido por especialidad es la suma de las duraciones de las operaciones por especialidad

    totalRequiredTime = 0

    for specialty in specialties:
        totalRequiredTime += requiredTime[specialty]            # tiempo total es la suma de los tiempos requeridos en todas las especialidades

    N_F = np.sum([isFriday(t) for t in times])                  # cantidad de viernes en el periodo de planeacion


    pabellon_disponible_AM = {}
    pabellon_disponible_PM = {}
    dias_completos = []
    for t in times:
        pabellon_disponible_AM[t] = 1
        pabellon_disponible_PM[t] = 1
        if q_AM(t) == 0:
            pabellon_disponible_AM[t] = 0
        if q_PM(t) == 0:
            pabellon_disponible_PM[t] = 0
        elif q_AM(t) > 0:
            dias_completos.append(t)

    specialties3hrs = []
    specialties5hrs = []
    specialties3y5 = []
    specialties_new = specialties
    if len(dias_completos) >= 4:
        for t in dias_completos:
            pabellones = R
            for s in specialties:
                cola = queue[queue["Service"] == s]
                Max_Duracion = cola[cola["Waiting_Time"] == max(cola["Waiting_Time"])]["MAIN_DURATION"][0]
                if Max_Duracion >= 300 and pabellones > 0 and s not in specialties3y5:
                    specialties5hrs.append([s, t])
                    specialties3y5.append(s)
                    pabellones -= 1
                elif Max_Duracion >= 180 and Max_Duracion < 300 and pabellones > 0 and s not in specialties3y5:
                    specialties3hrs.append([s, t])
                    specialties3y5.append(s)
                    pabellones -= 1

        specialties_new = list(set(specialties) - set(specialties3y5))

    H = 0
    for t in times:
        H += R * (pabellon_disponible_AM[t] * q_AM(t) + pabellon_disponible_PM[t] * q_PM(t))

    print(totalRequiredTime)
    h = {s: H * requiredTime[s] / totalRequiredTime for s in specialties}       # Horas ofrecibles de acuerdo a proporciones de tiempos requeridos por especialidad

    l = list(h.items())                         # transforma diccionario h en lista de tuplas (key,value)
    l = sorted(l, key=lambda z: z[1])           # y la ordena respecto de value


    m = {}
    r = 0
    for couple in l:
        s = couple[0]
        m[s] = r ** 2
        r += 1

    # No priority

    w = {}

    for s in specialties:
        w[s] = 1

    model = plp.LpProblem(name="Operation Room Scheduling")

    N = {}

    # para cada especialidad y dia se ponen cotas a las variables de decision
    for s in specialties:
        for t in times:
            N[s, t] = R         # todos los pabellones si quieren

    b_AM = {(s, t): plp.LpVariable(cat='Integer', lowBound=0, upBound=N[s, t], name='b_AM_{0}_{1}'.format(s, t))
            for s in specialties for t in times}

    b_PM = {(s, t): plp.LpVariable(cat='Integer', lowBound=0, upBound=N[s, t], name='b_PM_{0}_{1}'.format(s, t))
            for s in specialties for t in times}

    def add_constr(model, constraint):
        model.addConstraint(constraint)
        return constraint

    def isGranularity(n):                   ### AQUI esto habra que cambiarlo de acuerdo a las horas por turno ingresadas por usuario?
        leftovers = n % 5
        if (leftovers % 3) == 0:
            return True
        else:
            leftovers = n % 3

            if leftovers % 5 == 0:
                return True
            else:
                return False

    def bounds(n):
        lower = math.floor(n)
        upper = math.floor(n) + 1
        while isGranularity(upper) == False:
            upper += 1
        while isGranularity(lower) == False:
            lower -= 1
        return lower, upper

    def almostBounds(n):
        lower = round(n) - 1
        upper = round(n) + 1
        while isGranularity(upper) == False:
            upper += 1
        while isGranularity(lower) == False:
            lower -= 1
        if lower < 0:
            lower = 0
        return lower, upper

    def pabellones_disponibles_AM(total_de_pabellones,t):
        if pabellon_disponible_AM[t]:
            return total_de_pabellones
        else:
            return 0

    def pabellones_disponibles_PM(total_de_pabellones,t):
        if pabellon_disponible_PM[t]:
            return total_de_pabellones
        else:
            return 0

    lowerBounds = {}
    upperBounds = {}

    for s in specialties:
        lower, upper = almostBounds(h[s])
        lowerBounds[s] = lower
        upperBounds[s] = upper

    roundLowSwitch = True

    if roundLowSwitch:
        roundLowConstr = {
            s: add_constr(model, plp.LpConstraint(e=plp.lpSum(q_AM(t) * b_AM[s, t] + q_PM(t) * b_PM[s, t] for t in times),
                                                  sense=plp.LpConstraintGE,
                                                  rhs=lowerBounds[s],
                                                  name="roundLow_{0}".format(s)))
            for s in specialties_new}

    roundUpSwitch = True

    if roundUpSwitch:
        roundUpConstr = {
            s: add_constr(model, plp.LpConstraint(e=plp.lpSum(q_AM(t) * b_AM[s, t] + q_PM(t) * b_PM[s, t] for t in times),
                                                  sense=plp.LpConstraintLE,
                                                  rhs=upperBounds[s],
                                                  name="roundUp_{0}".format(s)))
            for s in specialties_new}

    sumAMConstr = {t: add_constr(model, plp.LpConstraint(e=plp.lpSum(b_AM[s, t] for s in specialties),
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=pabellones_disponibles_AM(R,t),
                                                         name="SumAM_{0}".format(t)))
                   for t in times}

    sumPMConstr = {t: add_constr(model, plp.LpConstraint(e=plp.lpSum(b_PM[s, t] for s in specialties),
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=pabellones_disponibles_PM(R,t),
                                                         name="SumPM_{0}".format(t)))
                   for t in times}


    treshrsrestriccion = {i: add_constr(model, plp.LpConstraint(e=b_AM[specialties3hrs[i][0],
                                                                       specialties3hrs[i][1]],
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=1,
                                                         name="Restric3h_{0}".format(i)))
                   for i in range(len(specialties3hrs))}

    cincohrsrestriccion1 = {i: add_constr(model, plp.LpConstraint(e=b_AM[specialties5hrs[i][0],
                                                                        specialties5hrs[i][1]],
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=1,
                                                         name="Restric5h1_{0}".format(i)))
                   for i in range(len(specialties5hrs))}
    cincohrsrestriccion2 = {i: add_constr(model, plp.LpConstraint(e=b_PM[specialties5hrs[i][0],
                                                                        specialties5hrs[i][1]],
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=1,
                                                         name="Restric5h2_{0}".format(i)))
                   for i in range(len(specialties5hrs))}

    objective = plp.lpSum(w[s] * (q_AM(t) * b_AM[s, t] + q_PM(t) * b_PM[s, t]) for s in specialties for t in times)

    model.sense = plp.LpMaximize
    model.setObjective(objective)

    start_time = time.time()
    model.solve()
    elapsed_time = time.time() - start_time

    print(plp.LpStatus[model.status])

    blocks = pd.DataFrame(index=specialties,
                          columns=['% Required Time', 'Fractional hours', 'Offered AM blocks', 'Offered PM blocks',
                                   'Offered hours'])

    for s in specialties:
        blocks.at[s, '% Required Time'] = 100 * requiredTime[s] / totalRequiredTime
        blocks.at[s, 'Fractional hours'] = h[s]
        blocks.at[s, 'Offered AM blocks'] = plp.value(plp.lpSum(b_AM[s, t] for t in times))
        blocks.at[s, 'Offered PM blocks'] = plp.value(plp.lpSum(b_PM[s, t] for t in times))
        blocks.at[s, 'Offered hours'] = plp.value(plp.lpSum(q_AM(t) * b_AM[s, t] + q_PM(t) * b_PM[s, t] for t in times))

    schedules = {}

    for t in times:
        wday = weekday(t)
        schedule = pd.DataFrame(index=[wday + '_AM', wday + '_PM'], columns=rooms)

        j = 0 #numero sala
        for s in blocks.index:
            in5hr = 1
            if [s, t] in specialties5hrs and in5hr:
                schedule.iat[0, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j + 1, bloque='AM',
                                        file=file, initial_duration=((q_AM(t) + q_PM(t)) * 60),
                                        remaining_duration=((q_AM(t) + q_PM(t)) * 60),
                                        bloque_extendido=1)
                new_schedule.save()
                schedule.iat[1, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j + 1, bloque='PM',
                                        file=file, initial_duration=(0),
                                        remaining_duration=(0),
                                        bloque_extendido=1)
                new_schedule.save()
                j +=1

        J = j #primera sala no ocupada por dias completos

        i = 0 #indica mañana
        j = J #numero sala

        for s in blocks.index:
            b = round(b_AM[s, t].varValue) - ([s ,t] in specialties5hrs)
            while b > 0:
                schedule.iat[i, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j+1, bloque='AM',
                                        file=file, initial_duration=(q_AM(t)*60), remaining_duration=(q_AM(t)*60))
                new_schedule.save()
                b -= 1
                j += 1

        i = 1 #indica tarde
        j = J #numero sala

        for s in blocks.index:
            b = round(b_PM[s, t].varValue) - ([s ,t] in specialties5hrs)
            while b > 0:
                schedule.iat[i, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j+1, bloque='PM',
                                    file=file, initial_duration=(q_PM(t)*60), remaining_duration=(q_PM(t)*60))
                new_schedule.save()
                b -= 1
                j += 1

def ingresacion(file, schedule, lista, u):

    for i in lista:

        if i.duracion:
            for s in schedule:

                if (s.remaining_duration > i.duracion + u) and (s.especialidad == i.especialidad) \
                        and (i.schedule.all().count() == 0):
                    s.remaining_duration = s.remaining_duration - (i.duracion + u)
                    s.save()
                    i.schedule.add(s)

                elif False and (s.initial_duration < i.duracion + u) and (s.remaining_duration < i.duracion + u) \
                        and (s.especialidad == i.especialidad) and (s.bloque == 'AM') and  (i.schedule.all().count() == 0):

                    schedule_pm = schedule.filter(especialidad=i.especialidad, bloque='PM', room=s.room, day=s.day).first()
                    if schedule_pm and s.remaining_duration + schedule_pm.remaining_duration > i.duracion + u:
                        time_remaining = i.duracion - (s.remaining_duration + u)
                        s.remaining_duration = 0
                        s.save()
                        schedule_pm.remaining_duration = schedule_pm.remaining_duration - time_remaining
                        i.schedule.add(s, schedule_pm)

def ingresacion2(file, schedule, lista, u):

    for i in lista:

        if i.duracion:
            for s in schedule:

                if (s.remaining_duration >= i.duracion + u) and (s.especialidad == i.especialidad) \
                        and (i.schedule.all().count() == 0):
                    s.remaining_duration = s.remaining_duration - (i.duracion + u)
                    s.save()
                    i.schedule.add(s)
                    i.prioridad = 1
                    i.save()

                elif False and (s.initial_duration < i.duracion + u) and (s.remaining_duration < i.duracion + u) \
                        and (s.especialidad == i.especialidad) and (s.bloque == 'AM') and  (i.schedule.all().count() == 0):

                    schedule_pm = schedule.filter(especialidad=i.especialidad, bloque='PM', room=s.room, day=s.day).first()
                    if schedule_pm and s.remaining_duration + schedule_pm.remaining_duration > i.duracion + u:
                        time_remaining = i.duracion - (s.remaining_duration + u)
                        s.remaining_duration = 0
                        s.save()
                        schedule_pm.remaining_duration = schedule_pm.remaining_duration - time_remaining
                        i.schedule.add(s, schedule_pm)
                        i.prioridad = 1
                        i.save()

def assign_list(file, schedule, ingresos, ingresos_prioritarios):

    u = 15

    for s in schedule:
        s.ingreso_set.clear()
        s.remaining_duration = s.initial_duration
        s.save()

    ingresacion(file, schedule.order_by('remaining_duration'), ingresos_prioritarios, u)
    #ingresacion(file, schedule, ingresos, u)

def assign_list2(file, schedule, ingresos, ingresos_prioritarios):

    u = 15

    for s in schedule:
        s.ingreso_set.clear()
        s.remaining_duration = s.initial_duration
        s.save()
    ingresacion2(file, schedule.order_by('remaining_duration'), ingresos_prioritarios, u)
    ingresacion2(file, schedule.order_by('remaining_duration'), ingresos, u)


def assign_list3(file, schedule, ingresos, ingresos_prioritarios):

    u = 15

    for s in schedule:
        s.ingreso_set.clear()
        s.remaining_duration = s.initial_duration
        s.save()
    ingresacion2(file, schedule.order_by('remaining_duration'), ingresos_prioritarios, u)
    return max(schedule.order_by('remaining_duration').last().remaining_duration - u, 0)

def digito_verificador(rut):
    rut = str(rut)
    if '-' in rut:
        return rut
    else:
        reversed_digits = map(int, reversed(str(rut)))
        factors = cycle(range(2, 8))
        s = sum(d * f for d, f in zip(reversed_digits, factors))
        verif = (-s) % 11
        if verif == 10:
            return rut + '-k'
        else:
            return rut + '-' + str(verif)

def normalizar_texto(s):

    s0 = s.strip()
    s1 = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
        r"\1",
        normalize("NFD", s0), 0, re.I
    )

    # -> NFC
    s2 = normalize('NFC', s1)
    s3 = s2.upper().replace(" ", "_")

    return s3