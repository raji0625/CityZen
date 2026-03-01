from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import sqlite3
import math
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_CSV = os.path.join(ROOT, 'data/master_assets.csv')
DB_PATH  = os.path.join(BASE, 'city_zen.db')
FRONTEND = os.path.join(ROOT, 'frontend')


# ── DB SETUP ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS complaints (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        asset_id      TEXT,
        asset_name    TEXT,
        reporter_name TEXT,
        contact       TEXT,
        severity      TEXT,
        description   TEXT,
        assigned_to   TEXT,
        status        TEXT DEFAULT 'Open',
        admin_note    TEXT,
        submitted_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()


# ── HELPERS ───────────────────────────────────────────────────
def load_assets():
    df = pd.read_csv(DATA_CSV)
    records = df.to_dict(orient='records')
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


# ── SERVE FRONTEND ────────────────────────────────────────────
@app.route('/')
def serve_login():
    return send_from_directory(FRONTEND, 'login.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND, filename)


# ── ASSETS ────────────────────────────────────────────────────
@app.route('/api/assets', methods=['GET'])
def get_assets():
    assets = load_assets()
    t = request.args.get('type')
    s = request.args.get('status')
    if t: assets = [a for a in assets if a['type'] == t]
    if s: assets = [a for a in assets if a['health_status'] == s]
    return jsonify(assets)

@app.route('/api/summary', methods=['GET'])
def get_summary():
    assets = load_assets()
    return jsonify({
        'total':    len(assets),
        'healthy':  sum(1 for a in assets if a['health_status'] == 'Healthy'),
        'warning':  sum(1 for a in assets if a['health_status'] == 'Warning'),
        'critical': sum(1 for a in assets if a['health_status'] == 'Critical'),
    })

@app.route('/api/assets/<asset_id>', methods=['GET'])
def get_asset(asset_id):
    asset = next((a for a in load_assets() if a['asset_id'] == asset_id), None)
    if asset:
        return jsonify(asset)
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/assets/save', methods=['POST'])
def save_asset():
    """Save a new asset from the anomaly checker (admin only)."""
    data = request.get_json()
    df = pd.read_csv(DATA_CSV)
    # Generate a new asset ID
    type_prefix = {'bridge': 'BR', 'building': 'BL', 'streetlight': 'SL',
                   'pipeline': 'PL', 'road': 'RD'}.get(data.get('type', ''), 'AS')
    existing = df[df['asset_id'].str.startswith(type_prefix)]
    new_id = f"{type_prefix}_{str(len(existing)+1).zfill(3)}"
    data['asset_id'] = new_id
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_CSV, index=False)
    return jsonify({'success': True, 'asset_id': new_id})


# ── ADMIN OVERRIDE ────────────────────────────────────────────
@app.route('/api/assets/<asset_id>/override', methods=['PATCH'])
def override_asset(asset_id):
    """Admin can directly update any asset's health status and sensor values."""
    data = request.get_json()
    df = pd.read_csv(DATA_CSV)
    idx = df[df['asset_id'] == asset_id].index
    if len(idx) == 0:
        return jsonify({'error': 'Asset not found'}), 404
    for key, val in data.items():
        if key in df.columns:
            df.at[idx[0], key] = val
    df.to_csv(DATA_CSV, index=False)
    return jsonify({'success': True})


# ── COMPLAINTS ────────────────────────────────────────────────
@app.route('/api/complaints', methods=['GET'])
def get_complaints():
    asset_id = request.args.get('asset_id')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if asset_id:
        rows = conn.execute('SELECT * FROM complaints WHERE asset_id=? ORDER BY submitted_at DESC', (asset_id,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM complaints ORDER BY submitted_at DESC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/complaints', methods=['POST'])
def add_complaint():
    d = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO complaints
        (asset_id, asset_name, reporter_name, contact, severity, description, assigned_to, status)
        VALUES (?,?,?,?,?,?,?,?)''',
        (d.get('asset_id'), d.get('asset_name'), d.get('reporter_name'),
         d.get('contact'), d.get('severity'), d.get('description'),
         d.get('assigned_to'), 'Open'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/complaints/<int:complaint_id>', methods=['PATCH'])
def update_complaint(complaint_id):
    d = request.get_json()
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE complaints SET status=?, admin_note=? WHERE id=?',
                 (d.get('status'), d.get('admin_note'), complaint_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/complaints/stats', methods=['GET'])
def complaint_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT status, COUNT(*) as count FROM complaints GROUP BY status').fetchall()
    conn.close()
    return jsonify({r['status']: r['count'] for r in rows})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)