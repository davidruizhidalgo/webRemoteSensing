from django.contrib import admin
from .models import DataSet
# Register your models here.
class DataSetAdmin(admin.ModelAdmin):
    readonly_fields = ("created","updated")

admin.site.register(DataSet, DataSetAdmin)