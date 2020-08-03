from django.db import models

# Create your models here.

class Website_vid(models.Model):
	
	inputVid = models.FileField(upload_to='', null=True, blank=True)
	outputImg = models.FileField(upload_to='', null=True, blank=True)
