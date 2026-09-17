const $ = (s, r=document) => r.querySelector(s);
let CFG = {backends:{}, has_gpu:false, gpu_name:"", defaultPrompt:""};
let settingsTab = null;
let selectedSource = null;

async function boot(){
  CFG = await (await fetch('/api/config')).json();
  try { CFG.defaultPrompt = (await (await fetch('/api/prompt')).json()).prompt || ''; } catch(e){ CFG.defaultPrompt = ''; }
  $('#gpu').textContent = CFG.gpu_name ? 'GPU: ' + CFG.gpu_name : 'GPU: none (MAI only for .srt)';
  initTheme();
  $('#theme').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  document.querySelectorAll('.nav[data-view]').forEach(b => b.onclick = () => show(b.dataset.view));
  renderRun(); renderSettings(); renderDocs(); show('run');
}

function initTheme(){ applyTheme(localStorage.getItem('theme') || 'dark'); }
function applyTheme(t){
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('theme', t); } catch(e){}
  $('#theme').textContent = t === 'dark' ? '\u2600' : '\u263D';   // show the mode you switch TO
  $('#theme').title = t === 'dark' ? 'Switch to light' : 'Switch to dark';
}

async function show(v){
  document.querySelectorAll('.view').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.nav[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view===v));
  $('#view-'+v).classList.remove('hidden');
  if(v==='jobs') await renderJobs();
}

function renderRun(){
  const opts = Object.entries(CFG.backends).map(([k,b]) => `<option value="${k}">${b.label}</option>`).join('');
  $('#view-run').innerHTML = `
    <h2>Run</h2>
    <div class="card">
      <label>Audio or video file</label>
      <div id="drop" class="dropzone" onclick="$('#fileInput').click()" ondragover="dzOver(event)" ondragleave="dzLeave(event)" ondrop="dzDrop(event)">
        <input id="fileInput" type="file" accept="audio/*,video/*,.mp3,.m4a,.wav,.mp4,.mkv,.webm,.aac,.flac" style="display:none" onchange="if(this.files[0])uploadFile(this.files[0])">
        Drag a file here, or <span class="link">click to choose</span>
        <div id="fmt" class="fmt"></div>
        <div id="picked" class="picked"></div>
      </div>
      <details style="margin-top:10px"><summary>Or pick from a folder on this machine (no copy - good for very large files)</summary>
        <div class="row" style="margin-top:8px"><input id="dir" placeholder="e.g. C:/audio"><button class="btn ghost" onclick="listFiles()">List</button></div>
        <div id="files" class="muted filelist" style="margin-top:8px">Enter a folder and click List.</div>
      </details>
    </div>
    <div class="card">
      <label>Backend</label>
      <select id="backend" onchange="onBackend()">${opts}</select>
      <div id="promptWrap">
        <label>Prompt sent to the model (edit if you want, or leave the default)</label>
        <textarea id="prompt"></textarea>
      </div>
      <label>Output</label>
      <div class="seg">
        <label><input type="radio" name="out" value="srt" checked> .srt</label>
        <label><input type="radio" name="out" value="txt"> .txt</label>
      </div>
      <div id="gate" class="muted" style="margin-top:8px"></div>
    </div>
    <button class="btn" onclick="startJob()">Run</button>`;
  $('#prompt').value = CFG.defaultPrompt;
  onBackend();
}

function setSource(path, label){ selectedSource = path; const p = $('#picked'); if(p) p.textContent = 'Selected: ' + label; }
function pickFolderFile(v){ setSource(v, v.split(/[\\/]/).pop()); }
async function uploadFile(f){
  const p = $('#picked'); if(p) p.textContent = 'Uploading ' + f.name + ' ...';
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/api/upload', {method:'POST', body:fd});
  if(!r.ok){ if(p) p.textContent = 'Upload failed.'; return; }
  const d = await r.json(); setSource(d.path, d.name);
}
function dzOver(e){ e.preventDefault(); $('#drop').classList.add('dragover'); }
function dzLeave(e){ $('#drop').classList.remove('dragover'); }
function dzDrop(e){ e.preventDefault(); $('#drop').classList.remove('dragover'); if(e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); }

async function listFiles(){
  const dir = $('#dir').value.trim();
  const r = await (await fetch('/api/files?dir='+encodeURIComponent(dir))).json();
  $('#files').innerHTML = r.files.length
    ? r.files.map(f => `<label><input type="radio" name="file" value="${f}" onchange="pickFolderFile(this.value)"> ${f}</label>`).join('')
    : 'No audio files found in that folder.';
}

function onBackend(){
  const b = CFG.backends[$('#backend').value];
  $('#promptWrap').style.display = b.has_prompt ? 'block' : 'none';
  const srtOk = !b.needs_gpu_for_srt || CFG.has_gpu;
  $('#gate').textContent = srtOk ? '' : 'This backend needs a GPU for .srt. Without one, pick .txt or the MAI backend.';
  if(!srtOk) document.querySelector('input[name=out][value=txt]').checked = true;
  const fmt = $('#fmt');
  if(fmt) fmt.innerHTML = b.video_ok
    ? 'Audio: mp3, m4a, wav, aac, flac&nbsp;&nbsp;|&nbsp;&nbsp;Video: mp4, mkv, webm (audio track is extracted)'
    : 'Audio only: mp3, m4a, wav, aac, flac, ogg (this backend does not accept video)';
  const fi = $('#fileInput');
  if(fi) fi.accept = b.video_ok ? 'audio/*,video/*,.mp3,.m4a,.wav,.mp4,.mkv,.webm,.aac,.flac'
                                : 'audio/*,.mp3,.m4a,.wav,.aac,.flac,.ogg';
}

async function startJob(){
  if(!selectedSource) return alert('Choose an audio file first (drop or click above, or pick from a folder).');
  const body = {backend:$('#backend').value, output:document.querySelector('input[name=out]:checked').value,
                source:selectedSource, prompt: $('#prompt') ? $('#prompt').value : ''};
  const r = await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){ return alert((await r.json()).error); }
  const {id} = await r.json();
  await show('jobs');
  openJob(id);
}

async function renderJobs(){
  const {jobs} = await (await fetch('/api/jobs')).json();
  const rows = jobs.length ? jobs.map(j => `<tr onclick="openJob('${j.id}')" style="cursor:pointer">
      <td><span class="chip ${j.status==='done'?'ok':''}">${j.status}</span></td>
      <td>${j.backend}</td><td>${j.output}</td><td>${(j.source||'').split(/[\\/]/).pop()}</td></tr>`).join('')
    : `<tr><td colspan="4" class="muted">No jobs yet. Start one from the Run tab.</td></tr>`;
  $('#view-jobs').innerHTML = `<h2>Jobs</h2><div class="card"><table><thead><tr>
    <th>Status</th><th>Backend</th><th>Output</th><th>File</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div id="jobDetail"></div>`;
}

function openJob(id){
  const detail = $('#jobDetail');
  if(!detail) return;
  detail.innerHTML = `<div class="card"><div class="row" style="justify-content:space-between">
    <div id="jstage" class="muted">stage: ...</div><button class="btn stop" onclick="stopJob('${id}')">Stop</button></div>
    <div id="results" style="margin:12px 0"></div><div class="log" id="log"></div></div>`;
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
  $('#results').innerHTML = files.length
    ? files.map(f => `<a class="btn ghost" href="/api/jobs/${id}/download?name=${encodeURIComponent(f)}">Download ${f.split('/').pop()}</a>`).join(' ')
    : '<span class="muted">No output files.</span>';
}
async function stopJob(id){ await fetch('/api/jobs/'+id+'/stop',{method:'POST'}); }

function renderSettings(){
  const keys = Object.keys(CFG.backends);
  if(!settingsTab || !keys.includes(settingsTab)) settingsTab = keys[0];
  const tabs = keys.map(k => `<button class="tab ${k===settingsTab?'active':''}" onclick="selectSettingsTab('${k}')">${CFG.backends[k].label}</button>`).join('');
  const b = CFG.backends[settingsTab];
  const fields = b.fields.map(f => `<label>${f.label}</label><input data-key="${f.key}" type="${f.secret?'password':'text'}" autocomplete="off">`).join('');
  $('#view-settings').innerHTML = `<h2>Settings</h2>
    <div class="tabs">${tabs}</div>
    <div class="card"><h3>${b.label}</h3>
      ${fields}
      <div class="row" style="margin-top:16px">
        <button class="btn" onclick="saveBackend()">Save</button>
        <button class="btn ghost" onclick="testBackend()">Test connection</button>
        <span id="test-msg" class="muted"></span>
      </div>
      <div class="muted" style="margin-top:10px">Keys are stored locally in .env and never leave this machine.</div>
    </div>`;
  loadSettings();
}
function selectSettingsTab(k){ settingsTab = k; renderSettings(); }
async function loadSettings(){
  const {values} = await (await fetch('/api/settings')).json();
  document.querySelectorAll('#view-settings input[data-key]').forEach(i => { if(values[i.dataset.key]) i.placeholder = values[i.dataset.key]; });
}
function collect(){
  const o = {};
  document.querySelectorAll('#view-settings input[data-key]').forEach(i => { if(i.value) o[i.dataset.key] = i.value; });
  return o;
}
async function saveBackend(){
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect())});
  $('#test-msg').textContent = 'Saved.';
}
async function testBackend(){
  $('#test-msg').textContent = 'Testing...';
  const r = await (await fetch('/api/test/'+settingsTab,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect())})).json();
  $('#test-msg').textContent = (r.ok ? 'OK - ' : 'FAILED - ') + r.message;
}

function renderDocs(){
  $('#view-docs').innerHTML = `<h2>How to use</h2>
    <div class="card doc">
      <h3>1. Configure a backend (Settings tab)</h3>
      <ol>
        <li>Open <b>Settings</b> and pick a backend tab: <b>MAI</b> (cheapest, needs no GPU), <b>Router</b>, or <b>Gemini (Vertex)</b>.</li>
        <li>Enter its API key / fields, click <b>Save</b>, then <b>Test connection</b>.</li>
      </ol>
      <h3>2. Run a job (Run tab)</h3>
      <ol>
        <li>Type the folder that holds your audio and click <b>List</b>, then pick a file.</li>
        <li>Choose the backend and the output: <code>.srt</code> (timed subtitles) or <code>.txt</code> (transcript only).</li>
        <li>Optional: edit the prompt sent to the model. Leave it as the default if unsure.</li>
        <li>Click <b>Run</b>.</li>
      </ol>
      <h3>3. Watch and download (Jobs tab)</h3>
      <ol>
        <li>The live log shows each stage. When the status is <b>done</b>, click <b>Download</b> to get the file.</li>
      </ol>
      <h3>Notes</h3>
      <ol>
        <li><code>.srt</code> from Router or Vertex needs an NVIDIA GPU (MMS alignment). <b>MAI produces <code>.srt</code> without a GPU</b> from its native word timestamps.</li>
        <li>This UI runs on <code>127.0.0.1</code> only - it is local to this machine.</li>
      </ol>
    </div>`;
}

boot();
