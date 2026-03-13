/**
 * app.js
 * Loads leaderboard.csv and renders the leaderboard table dynamically.
 */

const MEDALS = ['🥇', '🥈', '🥉'];

async function loadLeaderboard() {
  try {
    const response = await fetch('leaderboard.csv');
    const text     = await response.text();
    const entries  = parseCSV(text);
    renderLeaderboard(entries);
    renderStats(entries);
  } catch (err) {
    document.getElementById('leaderboard-body').innerHTML =
      '<tr><td colspan="7" class="empty">No submissions yet. Be the first to submit! 🚀</td></tr>';
  }
}

function parseCSV(text) {
  const lines   = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj    = {};
    headers.forEach((h, i) => obj[h.trim()] = values[i]?.trim());
    return obj;
  }).filter(e => e.team);
}

function renderLeaderboard(entries) {
  const tbody = document.getElementById('leaderboard-body');

  if (entries.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="7" class="empty">No submissions yet. Be the first to submit! 🚀</td></tr>';
    return;
  }

  // Sort by F1 descending
  entries.sort((a, b) => parseFloat(b.f1_score) - parseFloat(a.f1_score));
  const best = parseFloat(entries[0].f1_score);

  tbody.innerHTML = entries.map((e, i) => {
    const rank    = i + 1;
    const medal   = MEDALS[i] || `${rank}`;
    const f1      = parseFloat(e.f1_score);
    const barPct  = best > 0 ? (f1 / best * 100).toFixed(1) : 0;
    const rowClass = rank <= 3 ? `rank-${rank}` : '';

    return `
      <tr class="${rowClass}">
        <td class="rank-cell">${medal}</td>
        <td class="team-name">${escapeHtml(e.team)}</td>
        <td>
          <div class="score-bar-wrap">
            <span class="score-main">${f1.toFixed(4)}</span>
            <div class="score-bar" style="width:${barPct}px; max-width:80px;"></div>
          </div>
        </td>
        <td class="score">${parseFloat(e.accuracy).toFixed(4)}</td>
        <td class="score">${parseFloat(e.precision).toFixed(4)}</td>
        <td class="score">${parseFloat(e.recall).toFixed(4)}</td>
        <td class="date">${e.submitted_at || '—'}</td>
      </tr>
    `;
  }).join('');
}

function renderStats(entries) {
  document.getElementById('stat-teams').textContent   = entries.length;

  if (entries.length > 0) {
    const sorted = [...entries].sort((a, b) => parseFloat(b.f1_score) - parseFloat(a.f1_score));
    document.getElementById('stat-best').textContent    = parseFloat(sorted[0].f1_score).toFixed(4);
    document.getElementById('stat-leader').textContent  = sorted[0].team;
    document.getElementById('stat-updated').textContent = sorted[0].submitted_at?.split(' ')[0] || '—';
  } else {
    document.getElementById('stat-best').textContent    = '—';
    document.getElementById('stat-leader').textContent  = '—';
    document.getElementById('stat-updated').textContent = '—';
  }
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Auto-refresh every 60 seconds
loadLeaderboard();
setInterval(loadLeaderboard, 60000);
