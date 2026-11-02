# SPDX-License-Identifier: Apache-2.0
"""Application-layer denial proof for Harbor's transparent GOST boundary."""
import datetime
import json
import shlex
from urllib.parse import urlsplit

SERVICE = 'harbor-docker-egress-control-sidecar'
TARGETS = [('1.1.1.1',443), ('8.8.8.8',53)]
SANDBOX_PROBE = r'''
import json,socket,ssl,sys,urllib.request
out={'probes':[]}
for host,port in [('1.1.1.1',443),('8.8.8.8',53)]:
 row={'host':host,'port':port,'tcp_connected':False,'application_response':False}
 try:
  s=socket.create_connection((host,port),timeout=8)
  row.update(tcp_connected=True,source_port=s.getsockname()[1])
  if port==443:
   c=ssl.create_default_context();c.check_hostname=False;c.verify_mode=ssl.CERT_NONE
   s=c.wrap_socket(s,server_hostname=host)
   row['tls_handshake_completed']=True
   s.sendall(b'HEAD / HTTP/1.1\r\nHost: 1.1.1.1\r\nConnection: close\r\n\r\n')
  else:
   q=bytes.fromhex('123401000001000000000000')+b'\x07example\x07invalid\x00\x00\x01\x00\x01'
   s.sendall(len(q).to_bytes(2,'big')+q)
  row['application_response']=bool(s.recv(256));s.close()
 except Exception as e:row['error_type']=type(e).__name__
 out['probes'].append(row)
if sys.argv[1]:
 try:
  response=urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1],timeout=10)
  out['inference_health_status']=response.status
 except Exception as e:out['inference_health_error_type']=type(e).__name__
print(json.dumps(out))
'''


def denial_evidence(probe, log_rows):
    """No response is insufficient: require fresh GOST rejection of this socket."""
    if probe.get('application_response') or probe.get('tls_handshake_completed'):
        return None
    port=probe.get('source_port')
    if type(port) is not int or not probe.get('tcp_connected'):
        return None
    dst=f"{probe['host']}:{probe['port']}"
    matching=[r for r in log_rows if r.get('dst')==dst and str(r.get('client','')).endswith(':'+str(port))]
    for r in matching:
        sid=r.get('sid')
        session=[x for x in log_rows if sid and x.get('sid')==sid]
        if (r.get('inputBytes',0)>0 and r.get('outputBytes')==0
                and any(x.get('msg')=='bypass' for x in session)
                and not any(x.get('src') for x in session)):
            return {'session_id':sid,'destination':dst,'input_bytes':r['inputBytes'],
                    'output_bytes':0,'gost_rejection':'bypass','upstream_connection_observed':False}
    return None


async def audit(environment, inference_url=None):
    expected=[];health=''
    if inference_url:
        u=urlsplit(inference_url)
        if u.scheme not in ('http','https') or not u.hostname or u.username or u.password or u.query:
            raise RuntimeError('Isolation proof needs an uncredentialed verified inference endpoint')
        expected=[u.hostname];health=f'{u.scheme}://{u.netloc}/health'
    async def policy():
        show=await environment.service_exec(command='network-policy show',service=SERVICE,timeout_sec=10)
        hosts=await environment.service_exec(command='cat /opt/egress-sidecar/allowlist.txt',service=SERVICE,timeout_sec=10)
        actual=sorted(x.strip() for x in (hosts.stdout or '').splitlines() if x.strip())
        if show.return_code or hosts.return_code or 'mode: controlled egress' not in (show.stdout or '') or actual!=expected:
            raise RuntimeError('Isolation probe policy is not the exact declared inference allowlist')
        return {'controlled_egress':True,'allowed_hosts':actual}
    before=await policy()
    since=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=await environment.exec(command='python3 -c '+shlex.quote(SANDBOX_PROBE)+' '+shlex.quote(health),timeout_sec=35)
    if result.return_code!=0:
        raise RuntimeError('Isolation application probe could not execute; missing tools is not denial')
    try:data=json.loads(result.stdout)
    except (ValueError,TypeError) as e:raise RuntimeError('Isolation application probe returned invalid metadata') from e
    if [(p.get('host'),p.get('port')) for p in data.get('probes',[])]!=TARGETS:
        raise RuntimeError('Isolation probe target identity mismatch')
    # The native per-project compose executor scopes logs to this trial's sidecar.
    logs=await environment._run_docker_compose_command(['logs','--no-color','--no-log-prefix','--since',since,SERVICE],timeout_sec=10)
    if logs.return_code!=0:raise RuntimeError('Isolation sidecar rejection evidence unavailable')
    rows=[]
    for line in (logs.stdout or '').splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict):rows.append(r)
    after=await policy()
    for probe in data['probes']:
        proof=denial_evidence(probe,rows);probe['denial_evidence']=proof;probe['denied']=proof is not None
    return {'required_policy':'no-network','effective_policy':'inference-host allowlist' if inference_url else 'no-network',
            'scope':'Application-layer public TLS/HTTP and TCP-DNS denial with same-socket native GOST rejection; raw TCP can terminate at local proxy',
            'policy_before':before,'policy_after':after,**data,
            'ok':all(p['denied'] for p in data['probes']) and (not inference_url or data.get('inference_health_status')==200)}
