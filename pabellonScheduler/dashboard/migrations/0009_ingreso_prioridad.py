# Generated by Django 2.0 on 2019-07-29 15:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0008_ingreso_duracion'),
    ]

    operations = [
        migrations.AddField(
            model_name='ingreso',
            name='prioridad',
            field=models.IntegerField(null=True),
        ),
    ]
