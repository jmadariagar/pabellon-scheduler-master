# Programador de Pabellón

## Instalación

### Requerimientos

- Python 3.6

### Desarrollo

1. Clonar repositorio ```git clone https://github.com/caracena/pabellon-scheduler.git ```
2. Entrar a carpeta pabellonScheduler 
3. Ejecutar ```pip install -r requirements.txt``` para instalar dependencias
4. Ejecutar ``` python manage.py makemigrations ``` para crear base de datos
5. Ejecutar ``` python manage.py migrate ``` para aplicar cambios en base de datos
6. Ejecutar ``` python manage.py runserver ``` para correr aplicación en http://localhost:8000/ 


### Producción HLCM

1. Descargar zip desde la página del proyecto https://github.com/caracena/pabellon-scheduler 
2. Conectar a VPN del HLCM (preguntar a Nicolás González nicolasgonzalez@calvomackenna.cl por acceso)
3. Enviar zip del proyecto al servidor ```scp pabellon-scheduler-master.zip  claudioaracena@10.7.33.21:~ ```
4. Contectar por ssh al servidor ```ssh claudioaracena@10.7.33.21```. Password: claudioaracena.
4. Descomprimir el zip de proyecto ```unzip pabellon-scheduler-master.zip```. Si la carpeta del proyecto está creada, eliminar la carpeta con ```sudo rm -rf  pabellon-scheduler-master``` (Esto borra la base de datos igualmente) y luego descomprimir el zip del proyecto.
5. Entrar en la carpeta del proyecto (zip descomprimido)
6. Ejecutar ``` sudo docker-compose up -d ``` para correr aplicación en http://10.7.33.21:8086. Password: claudioaracena

Aqui va a haber problemas. Cuando haya que hacerlo, preferible hacerlo con Claudio
