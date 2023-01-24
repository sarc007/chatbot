import httplib2
from googleapiclient import discovery
from oauth2client.client import OAuth2Credentials as creds
crm = discovery.build(
    'cloudresourcemanager', 'v3', http=creds.authorize(httplib2.Http()))

operation = crm.projects().create(
body={
    'project_id': flags.projectId,
    'name': 'my project'
}).execute()
