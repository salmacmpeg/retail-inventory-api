const API_BASE = window.location.origin;

// ─── Auth & Setup ─────────────────────────────────────────
function getToken() {
    return localStorage.getItem('access_token');
}

function getUser() {
    try {
        return JSON.parse(localStorage.getItem('user') || '{}');
    } catch (e) {
        return {};
    }
}

function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    window.location.href = '/frontend/pages/login.html';
}

function requireAuth() {
    if (!getToken()) {
        window.location.href = '/frontend/pages/login.html';
        return false;
    }
    return true;
}

// ─── API Helper ────────────────────────────────────────────
function apiCall(path, method, body) {
    method = method || 'GET';
    var opts = {
        method: method,
        headers: {
            'Authorization': 'Bearer ' + getToken(),
            'Content-Type': 'application/json'
        }
    };
    if (body) {
        opts.body = JSON.stringify(body);
    }

    return fetch(API_BASE + path, opts).then(function(res) {
        return res.json().then(function(data) {
            if (res.status === 401) {
                logout();
                throw new Error('Session expired');
            }
            if (!res.ok) {
                throw new Error(data.detail || 'Request failed (' + res.status + ')');
            }
            return data;
        });
    });
}

// ─── Generic Table Renderer ───────────────────────────────
function renderTable(containerId, rows, columns, options) {
    options = options || {};
    var container = document.getElementById(containerId);
    if (!rows || !rows.length) {
        container.innerHTML = '<div class="empty">No data available</div>';
        return;
    }

    var html = '<table><thead><tr>';
    for (var i = 0; i < columns.length; i++) {
        html += '<th>' + columns[i].label + '</th>';
    }
    if (options.actions) {
        html += '<th>Actions</th>';
    }
    html += '</tr></thead><tbody>';

    for (var r = 0; r < rows.length; r++) {
        var row = rows[r];
        html += '<tr>';
        for (var c = 0; c < columns.length; c++) {
            var col = columns[c];
            var val = row[col.key];
            if (col.format) {
                val = col.format(val);
            } else if (typeof val === 'number' && col.key.indexOf('price') !== -1) {
                val = '$' + val.toFixed(2);
            } else if (col.badge) {
                var cls = 'badge-blue';
                if (val === 'delivered' || val === true || val === 'admin') {
                    cls = 'badge-green';
                } else if (val === 'shipped') {
                    cls = 'badge-blue';
                } else if (val === 'pending') {
                    cls = 'badge-yellow';
                } else if (val === 'customer') {
                    cls = 'badge-blue';
                } else if (val === 'inactive' || val === false) {
                    cls = 'badge-red';
                }
                var display = val === true ? 'Active' : val === false ? 'Inactive' : val;
                val = '<span class="badge ' + cls + '">' + display + '</span>';
            }
            html += '<td>' + (val != null ? val : '-') + '</td>';
        }
        if (options.actions) {
            html += '<td class="actions">';
            for (var a = 0; a < options.actions.length; a++) {
                var act = options.actions[a];
                if (act.condition && !act.condition(row)) {
                    continue;
                }
                var btnClass = act.class || 'btn btn-sm';
                var btnStyle = act.style || '';
                html += '<button class="' + btnClass + '" style="' + btnStyle + '" data-row-idx="' + r + '" data-action-idx="' + a + '">' + act.label + '</button>';
            }
            html += '</td>';
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;

    if (options.actions) {
        var table = container.querySelector('table');
        if (table) {
            table.addEventListener('click', function(e) {
                var btn = e.target.closest('button[data-row-idx]');
                if (!btn) return;
                var rowIdx = parseInt(btn.getAttribute('data-row-idx'));
                var actionIdx = parseInt(btn.getAttribute('data-action-idx'));
                var action = options.actions[actionIdx];
                if (action && action.onClick) {
                    action.onClick(rows[rowIdx]);
                }
            });
        }
    }
}

// ─── UI Helpers ───────────────────────────────────────────
function showToast(msg, type) {
    type = type || 'success';
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast ' + type + ' show';
    setTimeout(function() {
        t.classList.remove('show');
    }, 3000);
}

function showModal(id) {
    document.getElementById(id).classList.add('show');
}

function hideModal(id) {
    document.getElementById(id).classList.remove('show');
}

function switchSection(name) {
    var sections = document.querySelectorAll('.section');
    for (var i = 0; i < sections.length; i++) {
        sections[i].classList.add('hidden');
    }
    document.getElementById(name).classList.remove('hidden');

    var navs = document.querySelectorAll('.nav-item');
    for (var i = 0; i < navs.length; i++) {
        navs[i].classList.remove('active');
    }
    var nav = document.querySelector('.nav-item[data-section="' + name + '"]');
    if (nav) nav.classList.add('active');

    var titleMap = {
        dashboard: 'Dashboard',
        products: 'Products',
        orders: 'Orders',
        customers: 'Customers',
        users: 'Users',
        ml: 'ML Price Suggester',
        shop: 'Shop',
        cart: 'Cart'
    };
    document.getElementById('pageTitle').textContent = titleMap[name] || name;
}

// ─── Dashboard Init ───────────────────────────────────────
function initDashboard(config) {
    if (!requireAuth()) return;

    var user = getUser();
    var nameElId = config.role === 'admin' ? 'adminName' : 'customerName';
    var nameEl = document.getElementById(nameElId);
    if (nameEl) {
        nameEl.textContent = user.username || user.useremail || config.role;
    }

    var navItems = document.querySelectorAll('.nav-item[data-section]');
    for (var i = 0; i < navItems.length; i++) {
        (function(item) {
            item.addEventListener('click', function() {
                var section = item.dataset.section;
                switchSection(section);
                if (config.loadFns[section]) {
                    config.loadFns[section]();
                }
            });
        })(navItems[i]);
    }

    document.getElementById('logoutBtn').addEventListener('click', logout);
    document.getElementById('logoutBtn2').addEventListener('click', logout);

    switchSection(config.defaultSection);
    if (config.loadFns[config.defaultSection]) {
        config.loadFns[config.defaultSection]();
    }
}