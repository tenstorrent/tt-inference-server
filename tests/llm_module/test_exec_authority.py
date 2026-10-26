import copy
import socket
import pytest
from agentic_owned_networks.exec_authority import validate_exec_authority, validate_rendered_exec_authority


def subset(tmp_path):
    return {'HostConfig':{'Privileged':False,'PidMode':'','IpcMode':'private','CapAdd':None,'Devices':[],'DeviceRequests':None,'SecurityOpt':None,'Binds':[str(tmp_path)+':/logs:rw']},'Mounts':[{'Type':'bind','Source':str(tmp_path),'Destination':'/logs','RW':True,'Propagation':'rprivate'}]}


def test_normal_task_log_and_readonly_helper(tmp_path):
    item=subset(tmp_path);helper=tmp_path/'supervisor.py';helper.write_text('pass')
    item['Mounts'].append({'Type':'bind','Source':str(helper),'Destination':'/opt/tt-agentic-owned/exec_supervisor.py','RW':False,'Propagation':'rprivate'})
    assert validate_exec_authority(item)['mounts_checked']==2
    graph={'services':{'main':{'volumes':[{'type':'bind','source':str(tmp_path),'target':'/logs'},{'type':'bind','source':str(helper),'target':'/opt/tt-agentic-owned/exec_supervisor.py','read_only':True}]}}}
    assert validate_rendered_exec_authority(graph)['services_checked']==1


@pytest.mark.parametrize('key,value',[('Privileged',True),('PidMode','host'),('PidMode','container:abc'),('IpcMode','host'),('CapAdd',['SYS_ADMIN']),('CapAdd',['ALL']),('Devices',[{}]),('SecurityOpt',['apparmor=unconfined'])])
def test_dangerous_hostconfig(tmp_path,key,value):
    item=subset(tmp_path);item['HostConfig'][key]=value
    with pytest.raises(ValueError):validate_exec_authority(item)


@pytest.mark.parametrize('source,target',[('/','/host'),('/var/run/docker.sock','/control'),('/run','/control'),('/var/lib/docker','/data'),('/tmp','/proc')])
def test_control_mounts(tmp_path,source,target):
    item=subset(tmp_path);item['Mounts'][0].update(Source=source,Destination=target)
    with pytest.raises(ValueError):validate_exec_authority(item)


def test_actual_socket_alias_and_missing_source(tmp_path):
    item=subset(tmp_path)
    with socket.socket(socket.AF_UNIX) as sock:
        control=tmp_path/'innocent';sock.bind(str(control));item['Mounts'][0]['Source']=str(control)
        with pytest.raises(ValueError,match='socket'):validate_exec_authority(item)
    item['Mounts'][0]['Source']=str(tmp_path/'missing')
    with pytest.raises(ValueError,match='unavailable'):validate_exec_authority(item)


def test_root_symlink_and_incomplete(tmp_path):
    item=subset(tmp_path);alias=tmp_path/'alias';alias.symlink_to('/',target_is_directory=True);item['Mounts'][0]['Source']=str(alias)
    with pytest.raises(ValueError):validate_exec_authority(item)
    del item['HostConfig']['Privileged']
    with pytest.raises(ValueError,match='incomplete'):validate_exec_authority(item)


@pytest.mark.parametrize('service',[{'pid':'host'},{'ipc':'container:x'},{'volumes':['/:/host:ro']},{'volumes':['named:/data']},{'volumes_from':['foreign']},{'use_api_socket':True},{'cap_add':['NET_ADMIN']},{'volumes':[{'type':'bind','source':'/tmp','target':'/data','bind':{'propagation':'rshared'}}]}])
def test_compose_escapes(service):
    with pytest.raises(ValueError):validate_rendered_exec_authority({'services':{'main':service}})
