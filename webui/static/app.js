const $ = (s, r=document) => r.querySelector(s);
let CFG = {backends:{}, has_gpu:false};

async function boot(){
  CFG = await (await fetch('/api/config')).json();
  $('#gpu').textContent = CFG.has_gpu ? 'GPU: available' : 'GPU: none (MAI only for .srt)';
  document.querySelectorAll('.nav[data-view]').forEach(b => b.onclick = () => show(b.dataset.view));
  renderRun(); renderSettings(); show('run');
}
function show(v){
  document.querySelectorAll('.view').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.nav[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view===v));
  $('#view-'+v).classList.remove('hidden');
  if(v==='jobs') renderJobs();
}
function backendOptions(){
  return Object.entries(CFG.backends).map(([k,b]) => `<option value="${k}">${b.label}</option>`).join('');
}
function renderRun(){
  $('#view-run').innerHTML = `
    <h2>Run</h2>
    <div class="card">
      <label>Audio folder on this machine</label>
      <div class="row"><input id="dir" placeholder="e.g. C:/audio"><button class="btn ghost" onclick="listFiles()">List</button></div>
      <div id="files" class="muted" style="margin-top:8px">Enter a folder and click List.</div>
    </div>
    <div class="card">
      <label>Backend</label>
      <select id="backend" onchange="onBackend()">${backendOptions()}</select>
      <div id="promptWrap"><label>Prompt override (optional)</label><textarea id="prompt" placeholder="Leave empty to use the default prompt"></textarea></div>
      <label>Output</label>
      <div class="seg"><label class="chip"><input type="radio" name="out" value="srt" checked> .srt</label>
      <label class="chip"><input type="radio" name="out" value="txt"> .txt</label></div>
      <div id="gate" class="muted"></div>
    </div>
    <button class="btn" onclick="startJob()">Run</button>`;
  onBackend();
}
async function listFiles(){
  const dir = $('#dir').value.trim();
  const r = await (await fetch('/api/files?dir='+encodeURIComponent(dir))).json();
  $('#files').innerHTML = r.files.length
    ? r.files.map(f => `<label style="display:block"><input type="radio" name="file" value="${f}"> ${f}</label>`).join('')
    : 'No audio files found.';
}
function onBackend(){
  const b = CFG.backends[$('#backend').value];
  $('#promptWrap').style.display = b.has_prompt ? 'block' : 'none';
  const srtOk = !b.needs_gpu_for_srt || CFG.has_gpu;
  $('#gate').textContent = srtOk ? '' : 'This backend needs a GPU for .srt. Without one, choose .txt or the MAI backend.';
  if(!srtOk){ document.querySelector('input[name=out][value=txt]').checked = true; }
}
async function startJob(){
  const file = document.querySelector('input[name=file]:checked');
  if(!file) return alert('Pick an audio file first.');
  const body = {backend:$('#backend').value, output:document.querySelector('input[name=out]:checked').value,
                source:file.value, prompt:$('#prompt') ? $('#prompt').value : ''};
  const r = await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){ return alert((await r.json()).error); }
  const {id} = await r.json();
  show('jobs'); openJob(id);
}
async function renderJobs(){
  const {jobs} = await (await fetch('/api/jobs')).json();
  $('#view-jobs').innerHTML = `<h2>Jobs</h2><div class="card"><table><thead><tr>
    <th>Status</th><th>Backend</th><th>Output</th><th>File</th></tr></thead><tbody>
    ${jobs.map(j => `<tr onclick="openJob('${j.id}')" style="cursor:pointer">
      <td><span class="chip ${j.status==='done'?'ok':''}">${j.status}</span></td>
      <td>${j.backend}</td><td>${j.output}</td><td>${(j.source||'').split(/[\\/]/).pop()}</td></tr>`).join('')}
    </tbody></table></div><div id="jobDetail"></div>`;
}
function openJob(id){
  $('#jobDetail').innerHTML = `<div class="card"><div class="row" style="justify-content:space-between">
    <div id="jstage" class="muted">stage: ...</div><button class="btn stop" onclick="stopJob('${id}')">Stop</button></div>
    <div id="results"></div><div class="log" id="log"></div></div>`;
  const es = new EventSource('/api/jobs/'+id+'/events');
  const log = $('#log');
  es.addEventListener('log', e => { log.textContent += JSON.parse(e.data)+'\n'; log.scrollTop = log.scrollHeight; });
  es.addEventListener('status', async e => {
    const s = JSON.parse(e.data);
    const j = await (await fetch('/api/jobs/'+id)).json();
    $('#jstage').textContent = 'stage: '+(j.stage||s)+'  ('+j.status+')';
    if(['done','error','stopped'].includes(j.status)){ es.close(); showResults(id, j); }
  });
}
function showResults(id, j){
  const files = (j.results && j.results.files) || [];
  $('#results').innerHTML = files.map(f =>
    `<a class="btn ghost" href="/api/jobs/${id}/download?name=${encodeURIComponent(f)}">Download ${f}</a>`).join(' ');
}
async function stopJob(id){ await fetch('/api/jobs/'+id+'/stop',{method:'POST'}); }
function renderSettings(){
  $('#view-settings').innerHTML = '<h2>Settings</h2>' + Object.entries(CFG.backends).map(([k,b]) => `
    <div class="card"><h3>${b.label}</h3>
    ${b.fields.map(f => `<label>${f.label}</label><input data-key="${f.key}" type="${f.secret?'password':'text'}">`).join('')}
    <div class="row" style="margin-top:12px"><button class="btn" onclick="saveBackend('${k}')">Save</button>
    <button class="btn ghost" onclick="testBackend('${k}')">Test</button><span id="test-${k}" class="muted"></span></div></div>`).join('');
  loadSettings();
}
async function loadSettings(){
  const {values} = await (await fetch('/api/settings')).json();
  document.querySelectorAll('#view-settings input[data-key]').forEach(i => { if(values[i.dataset.key]) i.placeholder = values[i.dataset.key]; });
}
function collect(k){
  const o = {};
  document.querySelectorAll(`#view-settings .card`)[Object.keys(CFG.backends).indexOf(k)]
    .querySelectorAll('input[data-key]').forEach(i => { if(i.value) o[i.dataset.key] = i.value; });
  return o;
}
async function saveBackend(k){
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect(k))});
  $('#test-'+k).textContent = 'saved';
}
async function testBackend(k){
  $('#test-'+k).textContent = 'testing...';
  const r = await (await fetch('/api/test/'+k,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect(k))})).json();
  $('#test-'+k).textContent = (r.ok?'OK - ':'FAIL - ')+r.message;
}
boot();
