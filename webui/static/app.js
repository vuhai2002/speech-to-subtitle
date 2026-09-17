const $ = (s, r=document) => r.querySelector(s);
let CFG = {backends:{}, has_gpu:false, gpu_name:"", defaultPrompt:""};
let settingsTab = null;
let selectedSource = null;
let curJob = {id: null, files: []};

async function boot(){
  CFG = await (await fetch('/api/config')).json();
  try { CFG.defaultPrompt = (await (await fetch('/api/prompt')).json()).prompt || ''; } catch(e){ CFG.defaultPrompt = ''; }
  $('#gpu').textContent = CFG.gpu_name ? 'GPU: ' + CFG.gpu_name : 'GPU: none (MAI only for .srt)';
  $('#gpu').title = $('#gpu').textContent;   // full name on hover when the badge truncates on narrow screens
  initTheme();
  $('#theme').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  document.querySelectorAll('.nav[data-view]').forEach(b => b.onclick = () => show(b.dataset.view));
  renderRun(); renderSettings(); renderDocs(); show('run');
}

const ICON_SUN = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const ICON_MOON = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
const ICON_EYE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const ICON_EYE_OFF = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c6.5 0 10 7 10 7a13.2 13.2 0 0 1-1.67 2.68M6.6 6.6A13.5 13.5 0 0 0 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6"/><path d="M1 1l22 22"/></svg>';
function initTheme(){ applyTheme(localStorage.getItem('theme') || 'dark'); }
function applyTheme(t){
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('theme', t); } catch(e){}
  $('#theme').innerHTML = t === 'dark' ? ICON_SUN : ICON_MOON;   // show the mode you switch TO
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
  $('#view-jobs').innerHTML = `<h2>Jobs</h2><div class="card"><div class="tablewrap"><table><thead><tr>
    <th>Status</th><th>Backend</th><th>Output</th><th>File</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <div id="jobDetail"></div>`;
}

function openJob(id){
  const detail = $('#jobDetail');
  if(!detail) return;
  detail.innerHTML = `<div class="card"><div class="row" style="justify-content:space-between">
    <div id="jstage" class="muted">stage: ...</div><button id="stopBtn" class="btn stop" onclick="stopJob('${id}')">Stop</button></div>
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
  const stop = $('#stopBtn'); if(stop) stop.style.display = 'none';   // job is finished; nothing to stop
  const files = (j.results && j.results.files) || [];
  curJob = {id, files};
  if(!files.length){ $('#results').innerHTML = '<span class="muted">No output files.</span>'; return; }
  const tabs = files.map((f, i) => `<button class="tab" data-ri="${i}" onclick="viewFile(${i})">${f.split('/').pop()}</button>`).join('');
  $('#results').innerHTML = `<div class="row" style="justify-content:space-between;align-items:flex-end;gap:12px">
      <div class="tabs" style="margin-bottom:0">${tabs}</div>
      <div class="row" style="gap:8px">
        <button id="copyBtn" class="btn ghost sm" onclick="copyPreview()">Copy</button>
        <a id="dlBtn" class="btn ghost sm" href="#" download>Download</a>
      </div>
    </div><pre id="preview" class="preview"></pre>`;
  viewFile(0);                                 // preview the first output automatically
}
async function copyPreview(){
  const pre = $('#preview'); const btn = $('#copyBtn'); if(!pre) return;
  try { await navigator.clipboard.writeText(pre.textContent); }
  catch(e){                                    // fallback when the clipboard API is unavailable
    const r = document.createRange(); r.selectNode(pre);
    const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
    try { document.execCommand('copy'); } catch(_){}
    sel.removeAllRanges();
  }
  if(btn){ btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); }
}
async function viewFile(i){
  const f = curJob.files[i]; if(f === undefined) return;
  document.querySelectorAll('#results .tab').forEach(t => t.classList.toggle('active', Number(t.dataset.ri) === i));
  const url = '/api/jobs/' + curJob.id + '/download?name=' + encodeURIComponent(f);
  const dl = $('#dlBtn'); if(dl) dl.href = url;
  const pre = $('#preview'); if(!pre) return;
  pre.textContent = 'Loading ' + f.split('/').pop() + ' ...';
  try { const r = await fetch(url); pre.textContent = await r.text(); }
  catch(e){ pre.textContent = '(could not load ' + f + ')'; }
}
async function stopJob(id){ await fetch('/api/jobs/'+id+'/stop',{method:'POST'}); }

function renderSettings(){
  const keys = Object.keys(CFG.backends);
  if(!settingsTab || !keys.includes(settingsTab)) settingsTab = keys[0];
  const tabs = keys.map(k => `<button class="tab ${k===settingsTab?'active':''}" onclick="selectSettingsTab('${k}')">${CFG.backends[k].label}</button>`).join('');
  const b = CFG.backends[settingsTab];
  const fields = b.fields.map(f => {
    const ph = f.placeholder ? ` placeholder="${f.placeholder}"` : '';
    const input = `<input data-key="${f.key}" type="${f.secret?'password':'text'}"${ph} autocomplete="off">`;
    if(!f.secret) return `<label>${f.label}</label>${input}`;
    return `<label>${f.label}</label><div class="pw">${input}` +
           `<button type="button" class="pw-toggle" title="Show" aria-label="Show key" onclick="togglePw(this)">${ICON_EYE}</button></div>`;
  }).join('');
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
function togglePw(btn){
  const inp = btn.parentElement.querySelector('input'); if(!inp) return;
  const reveal = inp.type === 'password';
  inp.type = reveal ? 'text' : 'password';
  btn.innerHTML = reveal ? ICON_EYE_OFF : ICON_EYE;
  const lbl = reveal ? 'Hide key' : 'Show key';
  btn.title = lbl; btn.setAttribute('aria-label', lbl);
}
async function loadSettings(){
  // Load saved values into the fields (edit in place). Empty fields keep their example
  // placeholder. A secret loads into its password field - shown as dots until the eye reveals it.
  const {values} = await (await fetch('/api/settings')).json();
  document.querySelectorAll('#view-settings input[data-key]').forEach(i => { const v = values[i.dataset.key]; if(v) i.value = v; });
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
        <li>Drag an audio or video file onto the drop zone (or click to choose). For a very large file, expand <b>"pick from a folder"</b> and select it in place - no copy.</li>
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
