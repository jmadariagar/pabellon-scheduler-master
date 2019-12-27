import pandas as pd
import numpy as np
from datetime import timedelta
import time
import re
import pickle
import pulp as plp
import math
from .models import Schedule

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
    Datos = Datos.dropna()

    ## crear identidad
    for row in Datos.index:

        code = Datos.at[row, 'PRESTA_MIN']
        if type(code) == str:
            a, b, c = code.split('-')
            code = int(a + b + c)
            Datos.at[row, 'PRESTA_MIN'] = code
        else:
            Datos.at[row, 'PRESTA_MIN'] = 0

        yearE = Datos.at[row, 'F_ENTRADA'].year
        monthE = Datos.at[row, 'F_ENTRADA'].month
        dayE = Datos.at[row, 'F_ENTRADA'].day

        identity = str(row) + str(Datos.at[row, 'RUN']) + str(yearE) + str(monthE) + str(dayE)
        Datos.at[row, 'ID'] = identity

        waiting_time = (programming_date - Datos.at[row, 'F_ENTRADA']).days
        Datos.at[row, 'Waiting_Time'] = waiting_time

    Datos.reset_index(drop=True, inplace=True)
    Datos.drop_duplicates(['ID'], inplace=True)

    file_name = 'parameters/Extracted Parameters'
    with open(file_name, 'rb') as file_object:
        parameters = pickle.load(file_object)

    parameters = parameters[['MAIN_DURATION']]
    Datos = Datos.merge(parameters, how='left', left_on='PRESTA_MIN', right_index=True)

    print(Datos.shape)

    return Datos, missingColumns


def run_model(queue, programming_date, file):

    file_name = 'parameters/Extracted Parameters'
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

    specialties = queue['Service'].unique()
    S = len(specialties)

    operations = parameters.index

    T = int(file.ndays)  # [integer] number of days for the planning horizon
    R = int(file.nrooms)

    weekdays = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
    starting_date = programming_date
    starting_day = weekdays[starting_date.weekday()]

    estimator = 'MAIN_DURATION'  # Estimation according to the main operation. Yet the only relevant to use.

    times = range(1, T + 1)
    rooms = range(1, R + 1)

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

    u = 15

    # Regular day
    q_AM = int(file.hoursam)
    q_PM = int(file.hourspm)
    q = q_AM + q_PM

    # Friday
    q_PMF = 2
    q_F = q_AM + q_PMF

    def q_PM_(t):

        if isFriday(t) == 1:
            return q_PMF
        else:
            return q_PM

    requiredTime = {}

    for specialty in specialties:
        subtable = queue[queue['Service'] == specialty]
        subtable['Duration'] = subtable.index
        subtable['Duration'] = subtable['Duration'].apply(duration)
        requiredTime[specialty] = np.sum(subtable['Duration'])

    totalRequiredTime = 0

    for specialty in specialties:
        totalRequiredTime += requiredTime[specialty]

    N_F = np.sum([isFriday(t) for t in times])

    H = R * q * T - (q_PM - q_PMF) * N_F

    h = {s: H * requiredTime[s] / totalRequiredTime for s in specialties}

    l = list(h.items())
    l = sorted(l, key=lambda z: z[1])

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

    for s in specialties:
        for t in times:
            N[s, t] = 2
            if h[s] == 0:
                N[s, t] = 0

    b_AM = {(s, t): plp.LpVariable(cat='Integer', lowBound=0, upBound=N[s, t], name='b_AM_{0}_{1}'.format(s, t))
            for s in specialties for t in times}

    b_PM = {(s, t): plp.LpVariable(cat='Integer', lowBound=0, upBound=N[s, t], name='b_PM_{0}_{1}'.format(s, t))
            for s in specialties for t in times}

    def add_constr(model, constraint):
        model.addConstraint(constraint)
        return constraint

    def isGranularity(n):

        alpha = n // 5
        leftovers = n % 5

        if leftovers % 3 == 0:
            return True

        else:

            beta = n // 3
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

    lowerBounds = {}
    upperBounds = {}

    for s in specialties:
        lower, upper = almostBounds(h[s])
        lowerBounds[s] = lower
        upperBounds[s] = upper

    roundLowSwitch = True

    if roundLowSwitch:
        roundLowConstr = {
        s: add_constr(model, plp.LpConstraint(e=plp.lpSum(q_AM * b_AM[s, t] + q_PM_(t) * b_PM[s, t] for t in times),
                                              sense=plp.LpConstraintGE,
                                              rhs=lowerBounds[s],
                                              name="roundLow_{0}".format(s)))
        for s in specialties}

    roundUpSwitch = True

    if roundUpSwitch:
        roundUpConstr = {
        s: add_constr(model, plp.LpConstraint(e=plp.lpSum(q_AM * b_AM[s, t] + q_PM_(t) * b_PM[s, t] for t in times),
                                              sense=plp.LpConstraintLE,
                                              rhs=upperBounds[s],
                                              name="roundUp_{0}".format(s)))
        for s in specialties}

    sumAMConstr = {t: add_constr(model, plp.LpConstraint(e=plp.lpSum(b_AM[s, t] for s in specialties),
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=R,
                                                         name="SumAM_{0}".format(t)))
                   for t in times}

    sumPMConstr = {t: add_constr(model, plp.LpConstraint(e=plp.lpSum(b_PM[s, t] for s in specialties),
                                                         sense=plp.LpConstraintEQ,
                                                         rhs=R,
                                                         name="SumPM_{0}".format(t)))
                   for t in times}

    objective = plp.lpSum(w[s] * (q_AM * b_AM[s, t] + q_PM_(t) * b_PM[s, t]) for s in specialties for t in times)

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
        blocks.at[s, 'Offered hours'] = plp.value(plp.lpSum(q_AM * b_AM[s, t] + q_PM_(t) * b_PM[s, t] for t in times))

    for s in specialties:
        print(s, plp.value(plp.lpSum(b_PM[s, t] for t in times)))

    schedules = {}

    for t in times:
        wday = weekday(t)
        schedule = pd.DataFrame(index=[wday + '_AM', wday + '_PM'], columns=rooms)

        i = 0
        j = 0

        for s in blocks.index:
            b = round(b_AM[s, t].varValue)
            while b > 0:
                schedule.iat[i, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j, bloque='AM',
                                        file=file, initial_duration=(q_AM*60), remaining_duration=(q_AM*60))
                new_schedule.save()
                b -= 1
                j += 1

        i = 1
        j = 0

        for s in blocks.index:
            b = round(b_PM[s, t].varValue)
            while b > 0:
                schedule.iat[i, j] = s
                new_schedule = Schedule(especialidad=s, day=wday, room=j, bloque='PM',
                                        file=file, initial_duration=(q_PM_(t)*60), remaining_duration=(q_PM_(t)*60))
                new_schedule.save()
                b -= 1
                j += 1

        schedules[t] = schedule

def assign_list(file, schedule, ingresos, ingresos_prioritarios):

    for s in schedule:
        s.ingreso_set.clear()
        s.remaining_duration = s.initial_duration
        s.save()

    for i in ingresos_prioritarios:

        if i.duracion:
            for s in schedule:

                if (s.remaining_duration > i.duracion + 15) and (s.especialidad == i.especialidad) \
                        and (i.schedule.all().count() == 0):
                    s.remaining_duration = s.remaining_duration - (i.duracion + 15)
                    s.save()
                    i.schedule.add(s)

                elif (s.initial_duration < i.duracion + 15) and (s.remaining_duration < i.duracion + 15) \
                        and (s.especialidad == i.especialidad) and (s.bloque == 'AM') and  (i.schedule.all().count() == 0):

                    schedule_pm = schedule.filter(especialidad=i.especialidad, bloque='PM', room=s.room, day=s.day).first()
                    if schedule_pm and s.remaining_duration + schedule_pm.remaining_duration > i.duracion + 15:
                        time_remaining = i.duracion - (s.remaining_duration + 15)
                        s.remaining_duration = 0
                        s.save()
                        schedule_pm.remaining_duration = schedule_pm.remaining_duration - time_remaining
                        i.schedule.add(s, schedule_pm)

    for i in ingresos:

        if i.duracion:
            for s in schedule:

                if (s.remaining_duration > i.duracion + 15) and (s.especialidad == i.especialidad) \
                        and (i.schedule.all().count() == 0):
                    s.remaining_duration = s.remaining_duration - (i.duracion + 15)
                    s.save()
                    i.schedule.add(s)

                elif (s.initial_duration < i.duracion + 15) and (s.remaining_duration < i.duracion + 15) \
                        and (s.especialidad == i.especialidad) and (s.bloque == 'AM') and  (i.schedule.all().count() == 0):

                    schedule_pm = schedule.filter(especialidad=i.especialidad, bloque='PM', room=s.room, day=s.day).first()
                    if schedule_pm and s.remaining_duration + schedule_pm.remaining_duration > i.duracion + 15:
                        time_remaining = i.duracion - (s.remaining_duration + 15)
                        s.remaining_duration = 0
                        s.save()
                        schedule_pm.remaining_duration = schedule_pm.remaining_duration - time_remaining
                        i.schedule.add(s, schedule_pm)

