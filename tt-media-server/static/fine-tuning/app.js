const BASE = '/v1/fine_tuning';
const API_KEY = 'your-secret-key';
const HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`
};

let selectedJobId = null;
let metricsInterval = null;
let nextAfter = 0;
let metricsSeries = {};

const PARAM_DISPLAY_NAMES = {
    dataset_loader: "Dataset Loader",
    sst2: "SST-2",
    dataset_max_sequence_length: "Max Sequence Length",
    batch_size: "Batch Size",
    learning_rate: "Learning Rate",
    num_epochs: "Number of Epochs",
    val_steps_freq: "Validation Steps Frequency",
    steps_freq: "Training Steps Frequency",
    dtype: "Data Type",
    lora_r: "LoRA Rank (r)",
    lora_alpha: "LoRA Alpha",
    lora_target_modules: "LoRA Target Modules",
    lora_task_type: "LoRA Task Type",
    ignored_index: "Ignored Index",
};

const CHART_COLORS = { train_loss: '#2980b9', val_loss: '#e74c3c' };

const $ = (id) => document.getElementById(id);

function displayParamName(key) {
    return PARAM_DISPLAY_NAMES[key] || key;
}

function statusBadge(status) {
    return `<span class="badge badge-${status}">${status}</span>`;
}

function formatTime(ts) {
    if (!ts) return '-';
    return new Date(ts * 1000).toISOString().slice(0, 19).replace('T', ' ');
}

async function loadOptions() {
    try {
        const [datasets, models] = await Promise.all([
            fetch(`${BASE}/datasets`, { headers: HEADERS }).then(r => r.json()),
            fetch(`${BASE}/models`, { headers: HEADERS }).then(r => r.json()),
        ]);

        const sel = $('dataset_loader');
        sel.innerHTML = '';
        for (const d of datasets.data) {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            sel.appendChild(opt);
        }

        const modelSel = $('model_select');
        modelSel.innerHTML = '';
        for (const m of models.data) {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            modelSel.appendChild(opt);
        }
        $('info-text').textContent = '';
    } catch (e) {
        $('info-text').textContent = 'Failed to load options: ' + e.message;
    }
}

async function submitJob(e) {
    e.preventDefault();
    const statusEl = $('form-status');
    statusEl.textContent = 'Submitting...';
    statusEl.className = '';

    const body = {
        dataset_loader: $('dataset_loader').value,
        dataset_max_sequence_length: parseInt($('dataset_max_sequence_length').value),
        batch_size: parseInt($('batch_size').value),
        learning_rate: parseFloat($('learning_rate').value),
        num_epochs: parseInt($('num_epochs').value),
        val_steps_freq: parseInt($('val_steps_freq').value),
        steps_freq: parseInt($('steps_freq').value),
        dtype: $('dtype').value,
        lora_r: parseInt($('lora_r').value),
        lora_alpha: parseInt($('lora_alpha').value),
        lora_target_modules: $('lora_target_modules').value.split(',').map(s => s.trim()).filter(Boolean),
        lora_task_type: $('lora_task_type').value,
    };

    try {
        const res = await fetch(`${BASE}/jobs`, {
            method: 'POST', headers: HEADERS, body: JSON.stringify(body)
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const job = await res.json();
        statusEl.textContent = `Job created: ${job.id}`;
        statusEl.className = 'success-msg';
        refreshJobs();
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.className = 'error-msg';
    }
}

async function refreshJobs() {
    try {
        const res = await fetch(`${BASE}/jobs`, { headers: HEADERS });
        const data = await res.json();
        const tbody = document.querySelector('#jobs-table tbody');
        tbody.innerHTML = '';

        const jobs = data.data || [];
        jobs.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

        for (const job of jobs) {
            const tr = document.createElement('tr');
            if (job.id === selectedJobId) tr.classList.add('selected');
            tr.innerHTML = `
                <td><code>${job.id}</code></td>
                <td>${job.model || '-'}</td>
                <td>${statusBadge(job.status)}</td>
                <td>${formatTime(job.created_at)}</td>
            `;
            tr.addEventListener('click', () => selectJob(job.id));
            tbody.appendChild(tr);
        }

        if (selectedJobId) {
            refreshJobDetail(selectedJobId);
        }
    } catch (e) {
        console.error('Failed to refresh jobs:', e);
    }
}

async function refreshJobDetail(jobId) {
    const detailDiv = $('job-detail');
    detailDiv.hidden = false;
    $('detail-id').textContent = jobId;

    try {
        const res = await fetch(`${BASE}/jobs/${jobId}`, { headers: HEADERS });
        const job = await res.json();

        const meta = $('detail-meta');
        const entries = [
            ['Status', statusBadge(job.status)],
            ['Model', job.model || '-'],
            ['Created', formatTime(job.created_at)],
            ['Completed', formatTime(job.completed_at)],
        ];
        if (job.error) entries.push(['Error', job.error]);
        if (job.request_parameters) {
            for (const [k, v] of Object.entries(job.request_parameters)) {
                entries.push([k, Array.isArray(v) ? v.join(', ') : v]);
            }
        }
        meta.innerHTML = entries.map(([k, v]) => `<dt>${displayParamName(k)}</dt><dd>${v}</dd>`).join('');

        const cancelBtn = $('cancel-btn');
        const cancellable = ['queued', 'in_progress'].includes(job.status);
        cancelBtn.hidden = !cancellable;
        cancelBtn.onclick = () => cancelJob(jobId);
    } catch (e) {
        $('detail-meta').innerHTML = `<dt>Error</dt><dd>${e.message}</dd>`;
    }
}

async function selectJob(jobId) {
    selectedJobId = jobId;
    refreshJobs();
    await refreshJobDetail(jobId);

    metricsSeries = {};
    $('metrics-heading').textContent = 'No metrics to show';
    $('metrics-chart').style.display = 'none';
    nextAfter = 0;
    if (metricsInterval) clearInterval(metricsInterval);
    pollMetrics(jobId);
    metricsInterval = setInterval(() => pollMetrics(jobId), 2000);
}

async function pollMetrics(jobId) {
    if (jobId !== selectedJobId) return;
    try {
        const res = await fetch(
            `${BASE}/jobs/${jobId}/metrics?after=${nextAfter}`, { headers: HEADERS }
        );
        const data = await res.json();

        for (const m of (data.data || [])) {
            if (!metricsSeries[m.metric_name]) metricsSeries[m.metric_name] = [];
            metricsSeries[m.metric_name].push({ step: m.global_step, value: m.value });
        }

        drawChart();
        nextAfter = data.next_after;
        if (data.is_final) {
            clearInterval(metricsInterval);
            metricsInterval = null;
            refreshJobs();
        }
    } catch (e) {
        console.error('Failed to poll metrics:', e);
    }
}

async function cancelJob(jobId) {
    try {
        await fetch(`${BASE}/jobs/${jobId}/cancel`, {
            method: 'POST', headers: HEADERS
        });
        selectJob(jobId);
        refreshJobs();
    } catch (e) {
        console.error('Failed to cancel job:', e);
    }
}

function drawChart() {
    const canvas = $('metrics-chart');
    const ctx = canvas.getContext('2d');
    const pad = { top: 30, right: 150, bottom: 50, left: 80 };
    const plotW = canvas.width - pad.left - pad.right;
    const plotH = canvas.height - pad.top - pad.bottom;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let allSteps = [], allValues = [];
    for (const pts of Object.values(metricsSeries)) {
        for (const p of pts) { allSteps.push(p.step); allValues.push(p.value); }
    }
    if (allSteps.length === 0) {
        $('metrics-heading').textContent = 'No metrics to show';
        canvas.style.display = 'none';
        return;
    }
    $('metrics-heading').textContent = 'Training Metrics';
    canvas.style.display = 'block';

    const minStep = 0, maxStep = Math.max(...allSteps);
    const minVal = 0, maxVal = Math.max(...allValues);
    const valPad = (maxVal - minVal) * 0.05 || 0.001;
    const stepRange = maxStep - minStep || 1;
    const valRange = (maxVal + valPad) - (minVal);

    const toX = (step) => pad.left + ((step - minStep) / stepRange) * plotW;
    const toY = (val) => pad.top + plotH - ((val - (minVal)) / valRange) * plotH;

    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    const gridLines = 6;
    for (let i = 0; i <= gridLines; i++) {
        const y = pad.top + (plotH / gridLines) * i;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
        const gridVal = (minVal) + valRange * (1 - i / gridLines);
        ctx.fillStyle = '#888';
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(gridVal.toFixed(3), pad.left - 8, y + 6);
    }

    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '20px sans-serif';
    ctx.textAlign = 'center';
    const stepTicks = 5;
    for (let i = 0; i <= stepTicks; i++) {
        const step = minStep + (stepRange / stepTicks) * i;
        const rounded = Math.round(step);
        ctx.fillText(rounded, toX(rounded), canvas.height - 12);
    }

    ctx.fillStyle = '#666';
    ctx.font = '22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Step', pad.left + plotW / 2, canvas.height - 2);

    let legendY = pad.top + 10;
    const legendX = canvas.width - pad.right + 20;
    for (const [name, pts] of Object.entries(metricsSeries)) {
        const color = CHART_COLORS[name] || '#999';
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        const sorted = [...pts].sort((a, b) => a.step - b.step);
        for (let i = 0; i < sorted.length; i++) {
            const x = toX(sorted[i].step), y = toY(sorted[i].value);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        for (const p of sorted) {
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(toX(p.step), toY(p.value), 4, 0, Math.PI * 2); ctx.fill();
        }

        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY, 16, 16);
        ctx.fillStyle = '#333';
        ctx.textAlign = 'left';
        ctx.font = '20px sans-serif';
        ctx.fillText(name, legendX + 22, legendY + 14);
        legendY += 28;
    }
}

$('job-form').addEventListener('submit', submitJob);
loadOptions();
refreshJobs();
setInterval(refreshJobs, 5000);