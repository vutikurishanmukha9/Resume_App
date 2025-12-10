/**
 * AI Resume Analyzer - Frontend Application
 * Optimized and Enhanced Version
 */

(function () {
    'use strict';

    // ==================== CONFIGURATION ====================
    const CONFIG = {
        MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
        ALLOWED_TYPES: ['application/pdf', 'text/plain'],
        MIN_JD_LENGTH: 20,
        ENDPOINTS: {
            UPLOAD: '/upload',
            MATCH: '/match_jd_resume',
            ATS_SCORE: '/ats_score'
        },
        SCROLL_OPTIONS: {
            behavior: 'smooth',
            block: 'nearest'
        }
    };

    // ==================== DOM CACHE ====================
    const DOM = {
        uploadForm: null,
        fileInput: null,
        fileName: null,
        analyzeBtn: null,
        matchBtn: null,
        jdText: null,
        result: null,
        jdMatchResult: null,
        atsResult: null,
        atsBtn: null,
        atsModeToggle: null,
        error: null,
        fileLabel: null,
        themeToggle: null
    };

    // ==================== THEME MANAGEMENT ====================
    /**
     * Initialize theme from localStorage or system preference
     */
    function initTheme() {
        const savedTheme = localStorage.getItem('theme');
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        } else if (!systemPrefersDark) {
            document.documentElement.setAttribute('data-theme', 'light');
        }
        // Default is dark (no attribute needed as :root is dark)
    }

    /**
     * Toggle between light and dark theme
     */
    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        if (newTheme === 'dark') {
            document.documentElement.removeAttribute('data-theme');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }

        localStorage.setItem('theme', newTheme);
    }

    // ==================== INITIALIZATION ====================
    /**
     * Initialize application when DOM is ready
     */
    function init() {
        // Initialize theme first (before anything renders)
        initTheme();

        // Cache DOM elements
        cacheDOMElements();

        // Check browser compatibility
        if (!checkBrowserCompatibility()) {
            showError('Your browser does not support required features. Please use a modern browser.');
            return;
        }

        // Attach event listeners
        attachEventListeners();

        // Theme toggle listener
        if (DOM.themeToggle) {
            DOM.themeToggle.addEventListener('click', toggleTheme);
        }

        console.log('AI Resume Analyzer initialized successfully');
    }

    /**
     * Cache all DOM elements for performance
     */
    function cacheDOMElements() {
        DOM.uploadForm = document.getElementById('uploadForm');
        DOM.fileInput = document.getElementById('resumeFile');
        DOM.fileName = document.getElementById('fileName');
        DOM.analyzeBtn = document.getElementById('analyzeBtn');
        DOM.matchBtn = document.getElementById('matchBtn');
        DOM.jdText = document.getElementById('jdText');
        DOM.result = document.getElementById('result');
        DOM.jdMatchResult = document.getElementById('jdMatchResult');
        DOM.atsResult = document.getElementById('atsResult');
        DOM.atsBtn = document.getElementById('atsBtn');
        DOM.atsModeToggle = document.getElementById('atsModeToggle');
        DOM.error = document.getElementById('error');
        DOM.fileLabel = document.querySelector('.file-label');
        DOM.themeToggle = document.getElementById('themeToggle');
    }

    /**
     * Check if browser supports required APIs
     */
    function checkBrowserCompatibility() {
        return !!(window.FormData && window.fetch && window.File);
    }

    // ==================== EVENT LISTENERS ====================
    /**
     * Attach all event listeners
     */
    function attachEventListeners() {
        // File input change
        DOM.fileInput.addEventListener('change', handleFileChange);

        // Form submission for analysis
        DOM.uploadForm.addEventListener('submit', handleAnalyze);

        // JD match button
        DOM.matchBtn.addEventListener('click', handleJDMatch);

        // ATS Score button
        if (DOM.atsBtn) {
            DOM.atsBtn.addEventListener('click', handleATSScore);
        }

        // Show ATS mode toggle when JD is provided
        if (DOM.jdText) {
            DOM.jdText.addEventListener('input', function () {
                if (DOM.atsModeToggle) {
                    DOM.atsModeToggle.style.display = this.value.trim().length > 0 ? 'flex' : 'none';
                }
            });
        }

        // Drag and drop
        setupDragAndDrop();

        // Keyboard shortcuts
        setupKeyboardShortcuts();

        // File label keyboard accessibility
        DOM.fileLabel.addEventListener('keydown', handleFileLabelKeydown);
    }

    /**
     * Handle file input change
     */
    function handleFileChange(e) {
        const file = e.target.files[0];

        if (!file) {
            clearFileDisplay();
            return;
        }

        // Validate file
        const validation = validateFile(file);

        if (!validation.valid) {
            showError(validation.error);
            clearFileInput();
            return;
        }

        // Display file info
        displayFileInfo(file);
        hideError();
    }

    /**
     * Handle resume analysis
     */
    async function handleAnalyze(e) {
        e.preventDefault();

        const file = DOM.fileInput.files[0];

        if (!file) {
            showError('Please select a resume file to upload.');
            return;
        }

        const validation = validateFile(file);
        if (!validation.valid) {
            showError(validation.error);
            return;
        }

        // Clear previous results
        resetResults();

        // Prepare form data
        const formData = new FormData();
        formData.append('resume', file);

        // Show loading state
        setButtonLoading(DOM.analyzeBtn, true);

        try {
            const response = await fetch(CONFIG.ENDPOINTS.UPLOAD, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error (${response.status})`);
            }

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || 'Failed to analyze resume. Please try again.');
            }

        } catch (error) {
            console.error('Analysis error:', error);
            handleFetchError(error, 'analyze');
        } finally {
            setButtonLoading(DOM.analyzeBtn, false);
        }
    }

    /**
     * Handle JD match
     */
    async function handleJDMatch() {
        const file = DOM.fileInput.files[0];
        const jdText = DOM.jdText.value.trim();

        // Validate inputs
        if (!file) {
            showError('Please select a resume file first.');
            DOM.fileInput.focus();
            return;
        }

        if (!jdText) {
            showError('Please paste a Job Description to check the match.');
            DOM.jdText.focus();
            return;
        }

        if (jdText.length < CONFIG.MIN_JD_LENGTH) {
            showError(`Job Description is too short. Please provide at least ${CONFIG.MIN_JD_LENGTH} characters.`);
            DOM.jdText.focus();
            return;
        }

        const validation = validateFile(file);
        if (!validation.valid) {
            showError(validation.error);
            return;
        }

        // Clear previous results
        resetResults();

        // Prepare form data
        const formData = new FormData();
        formData.append('resume', file);
        formData.append('jd_text', jdText);

        // Show loading state
        setButtonLoading(DOM.matchBtn, true);

        try {
            const response = await fetch(CONFIG.ENDPOINTS.MATCH, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error (${response.status})`);
            }

            if (data.success) {
                displayJDMatch(data);
            } else {
                showError(data.error || 'Failed to calculate JD match. Please try again.');
            }

        } catch (error) {
            console.error('JD match error:', error);
            handleFetchError(error, 'match');
        } finally {
            setButtonLoading(DOM.matchBtn, false);
        }
    }

    /**
     * Handle ATS Score calculation
     */
    async function handleATSScore() {
        const file = DOM.fileInput.files[0];
        const jdText = DOM.jdText.value.trim();

        // Validate inputs
        if (!file) {
            showError('Please select a resume file first.');
            DOM.fileInput.focus();
            return;
        }

        if (!jdText) {
            showError('Please paste a Job Description to calculate ATS score.');
            DOM.jdText.focus();
            return;
        }

        if (jdText.length < CONFIG.MIN_JD_LENGTH) {
            showError(`Job Description is too short. Please provide at least ${CONFIG.MIN_JD_LENGTH} characters.`);
            DOM.jdText.focus();
            return;
        }

        const validation = validateFile(file);
        if (!validation.valid) {
            showError(validation.error);
            return;
        }

        // Clear previous results
        resetResults();

        // Get selected mode
        const modeInput = document.querySelector('input[name="atsMode"]:checked');
        const mode = modeInput ? modeInput.value : 'deep';

        // Prepare form data
        const formData = new FormData();
        formData.append('resume', file);
        formData.append('jd_text', jdText);
        formData.append('mode', mode);

        // Show loading state
        setButtonLoading(DOM.atsBtn, true);

        try {
            const response = await fetch(CONFIG.ENDPOINTS.ATS_SCORE, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server error (${response.status})`);
            }

            if (data.success) {
                displayATSResult(data);
            } else {
                showError(data.error || 'Failed to calculate ATS score. Please try again.');
            }

        } catch (error) {
            console.error('ATS Score error:', error);
            handleFetchError(error, 'ats');
        } finally {
            setButtonLoading(DOM.atsBtn, false);
        }
    }

    /**
     * Display ATS Score results
     */
    function displayATSResult(data) {
        if (!data || typeof data.ats_score !== 'number') {
            showError('Invalid response from server. Please try again.');
            return;
        }

        const score = data.ats_score;
        const interpretation = data.interpretation || {};
        const subScores = data.sub_scores || {};
        const matchedKeywords = data.matched_keywords || [];
        const missingKeywords = data.missing_keywords || {};
        const achievementsFound = data.achievements_found || [];
        const suggestions = data.suggestions || [];
        const mode = data.mode || 'deep';

        // Determine score color
        let scoreColor = '';
        if (score >= 85) scoreColor = 'excellent';
        else if (score >= 70) scoreColor = 'good';
        else if (score >= 50) scoreColor = 'fair';
        else scoreColor = 'poor';

        let html = `
            <h2><i class="fas fa-clipboard-check"></i> ATS Score Analysis</h2>
            <div class="ats-mode-badge ${mode}">${mode === 'deep' ? 'Deep Analysis' : 'Quick Scan'}</div>
            
            <!-- Main Score Card -->
            <div class="ats-score-card ${scoreColor}">
                <div class="score-circle">
                    <svg viewBox="0 0 100 100">
                        <circle class="score-bg" cx="50" cy="50" r="45"></circle>
                        <circle class="score-fill" cx="50" cy="50" r="45" 
                            stroke-dasharray="${score * 2.83} 283"
                            stroke-dashoffset="0"></circle>
                    </svg>
                    <div class="score-value">${score}</div>
                </div>
                <div class="score-details">
                    <div class="score-badge ${scoreColor}">${escapeHtml(interpretation.badge || 'Score')}</div>
                    <div class="score-message">${escapeHtml(interpretation.message || '')}</div>
                </div>
            </div>
        `;

        // Sub-scores breakdown (only in deep mode)
        if (mode === 'deep' && Object.keys(subScores).length > 0) {
            html += `
                <div class="result-card">
                    <h3><i class="fas fa-chart-bar"></i> Score Breakdown</h3>
                    <div class="sub-scores">
            `;

            const subScoreLabels = {
                'skill_match': { label: 'Skill Match', icon: 'fa-cogs', weight: '40%' },
                'title_match': { label: 'Title Match', icon: 'fa-user-tie', weight: '20%' },
                'experience': { label: 'Experience', icon: 'fa-briefcase', weight: '15%' },
                'achievement': { label: 'Achievements', icon: 'fa-trophy', weight: '10%' },
                'education': { label: 'Education', icon: 'fa-graduation-cap', weight: '10%' },
                'formatting_penalty': { label: 'Format Penalty', icon: 'fa-exclamation-triangle', weight: '-5%' }
            };

            for (const [key, value] of Object.entries(subScores)) {
                const config = subScoreLabels[key] || { label: key, icon: 'fa-check', weight: '' };
                const isPenalty = key === 'formatting_penalty';
                const displayValue = isPenalty ? `-${value}` : value;

                html += `
                    <div class="sub-score-item ${isPenalty ? 'penalty' : ''}">
                        <div class="sub-score-header">
                            <span class="sub-score-label">
                                <i class="fas ${config.icon}"></i> ${config.label}
                            </span>
                            <span class="sub-score-value">${displayValue}%</span>
                        </div>
                        <div class="sub-score-bar">
                            <div class="sub-score-fill ${isPenalty ? 'penalty' : ''}" 
                                style="width: ${Math.abs(value)}%"></div>
                        </div>
                        <div class="sub-score-weight">${config.weight}</div>
                    </div>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        // Matched Keywords
        if (matchedKeywords.length > 0) {
            html += `
                <div class="result-card">
                    <h3><i class="fas fa-check-circle"></i> Matched Keywords (${matchedKeywords.length})</h3>
                    <div class="keywords-list matched">
            `;

            matchedKeywords.forEach(kw => {
                const keyword = typeof kw === 'object' ? kw.keyword : kw;
                const importance = typeof kw === 'object' ? kw.importance : 'standard';
                html += `<span class="keyword-badge matched ${importance}">${escapeHtml(keyword)}</span>`;
            });

            html += `
                    </div>
                </div>
            `;
        }

        // Missing Keywords
        if (missingKeywords.critical?.length || missingKeywords.important?.length || missingKeywords.optional?.length) {
            html += `
                <div class="result-card">
                    <h3><i class="fas fa-exclamation-circle"></i> Missing Keywords</h3>
                    <p class="section-description">Add these keywords to improve your ATS score:</p>
            `;

            if (missingKeywords.critical?.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title critical"><i class="fas fa-fire"></i> Critical</h4>
                        <div class="keywords-list">
                            ${missingKeywords.critical.map(kw => `
                                <span class="keyword-badge critical">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.important?.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title important"><i class="fas fa-star"></i> Important</h4>
                        <div class="keywords-list">
                            ${missingKeywords.important.map(kw => `
                                <span class="keyword-badge important">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.optional?.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title optional"><i class="fas fa-plus-circle"></i> Optional</h4>
                        <div class="keywords-list">
                            ${missingKeywords.optional.map(kw => `
                                <span class="keyword-badge optional">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Achievements Found
        if (achievementsFound.length > 0) {
            html += `
                <div class="result-card">
                    <h3><i class="fas fa-trophy"></i> Achievements Detected (${achievementsFound.length})</h3>
                    <ul class="achievements-list">
                        ${achievementsFound.map(achievement => `
                            <li><i class="fas fa-check"></i> ${escapeHtml(achievement)}</li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        // Suggestions
        if (suggestions.length > 0) {
            html += `
                <div class="result-card suggestions-card">
                    <h3><i class="fas fa-lightbulb"></i> Improvement Suggestions</h3>
                    <ul class="suggestions-list">
                        ${suggestions.map(suggestion => `
                            <li><i class="fas fa-arrow-right"></i> ${escapeHtml(suggestion)}</li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        DOM.atsResult.innerHTML = html;
        DOM.atsResult.style.display = 'block';
        scrollToElement(DOM.atsResult);
    }

    /**
     * Handle file label keyboard interaction
     */
    function handleFileLabelKeydown(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            DOM.fileInput.click();
        }
    }

    // ==================== DRAG AND DROP ====================
    /**
     * Setup drag and drop functionality
     */
    function setupDragAndDrop() {
        ['dragover', 'dragenter'].forEach(eventName => {
            DOM.fileLabel.addEventListener(eventName, handleDragOver);
        });

        ['dragleave', 'dragend'].forEach(eventName => {
            DOM.fileLabel.addEventListener(eventName, handleDragLeave);
        });

        DOM.fileLabel.addEventListener('drop', handleDrop);
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        DOM.fileLabel.classList.remove('drag-over');

        const files = e.dataTransfer.files;

        if (files.length > 0) {
            DOM.fileInput.files = files;
            const event = new Event('change', { bubbles: true });
            DOM.fileInput.dispatchEvent(event);
        }
    }

    // ==================== KEYBOARD SHORTCUTS ====================
    /**
     * Setup keyboard shortcuts
     */
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
            // Ctrl/Cmd + Enter to analyze
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (!DOM.analyzeBtn.disabled && DOM.fileInput.files.length > 0) {
                    e.preventDefault();
                    DOM.uploadForm.dispatchEvent(new Event('submit'));
                }
            }

            // Escape to reset
            if (e.key === 'Escape') {
                resetForm();
            }
        });
    }

    // ==================== VALIDATION ====================
    /**
     * Validate uploaded file
     */
    function validateFile(file) {
        if (!file) {
            return {
                valid: false,
                error: 'No file selected. Please choose a file to upload.'
            };
        }

        if (file.size === 0) {
            return {
                valid: false,
                error: 'The selected file is empty. Please choose a valid file.'
            };
        }

        if (file.size > CONFIG.MAX_FILE_SIZE) {
            return {
                valid: false,
                error: `File size (${formatFileSize(file.size)}) exceeds the 16MB limit. Please upload a smaller file.`
            };
        }

        if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
            return {
                valid: false,
                error: 'Invalid file type. Only PDF and TXT files are supported.'
            };
        }

        return { valid: true };
    }

    // ==================== UI CONTROL ====================
    /**
     * Set button loading state
     */
    function setButtonLoading(button, isLoading) {
        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');

        button.disabled = isLoading;
        btnText.style.display = isLoading ? 'none' : 'inline';
        btnLoader.style.display = isLoading ? 'inline-flex' : 'none';
    }

    /**
     * Reset all results
     */
    function resetResults() {
        DOM.result.style.display = 'none';
        DOM.result.innerHTML = '';
        DOM.jdMatchResult.style.display = 'none';
        DOM.jdMatchResult.innerHTML = '';
        if (DOM.atsResult) {
            DOM.atsResult.style.display = 'none';
            DOM.atsResult.innerHTML = '';
        }
        hideError();
    }

    /**
     * Reset entire form
     */
    function resetForm() {
        DOM.uploadForm.reset();
        clearFileDisplay();
        resetResults();
    }

    /**
     * Display file info
     */
    function displayFileInfo(file) {
        DOM.fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
        DOM.fileName.classList.add('active');
    }

    /**
     * Clear file display
     */
    function clearFileDisplay() {
        DOM.fileName.textContent = '';
        DOM.fileName.classList.remove('active');
    }

    /**
     * Clear file input
     */
    function clearFileInput() {
        DOM.fileInput.value = '';
        clearFileDisplay();
    }

    /**
     * Show error message
     */
    function showError(message) {
        DOM.error.innerHTML = `<strong>Error:</strong> ${escapeHtml(message)}`;
        DOM.error.style.display = 'block';
        scrollToElement(DOM.error);
    }

    /**
     * Hide error message
     */
    function hideError() {
        DOM.error.style.display = 'none';
        DOM.error.innerHTML = '';
    }

    /**
     * Handle fetch errors with user-friendly messages
     */
    function handleFetchError(error, action) {
        let message = '';

        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            message = 'Unable to connect to the server. Please check your internet connection and try again.';
        } else if (error.message.includes('503')) {
            message = 'The system is still initializing. Please wait a moment and try again.';
        } else if (error.message.includes('413')) {
            message = 'File size is too large. Please upload a file smaller than 16MB.';
        } else if (error.message.includes('429')) {
            message = '⏱️ Rate limit exceeded. You\'ve made too many requests. Please wait a minute and try again.';
        } else if (error.message.includes('500')) {
            message = 'A server error occurred. Please try again later.';
        } else {
            message = error.message || `An error occurred while ${action === 'analyze' ? 'analyzing the resume' : 'calculating the match'}. Please try again.`;
        }

        showError(message);
    }

    // ==================== DISPLAY RESULTS ====================
    /**
     * Display analysis results
     */
    function displayResults(data) {
        if (!data || !data.predicted_job || !data.matches) {
            showError('Invalid response from server. Please try again.');
            return;
        }

        let html = `
            <h2> Analysis Results</h2>
            <div class="result-card">
                <h3> Predicted Job Category</h3>
                <div class="job-badge">${escapeHtml(data.predicted_job)}</div>
            </div>
            
            <div class="result-card">
                <h3> Top Job Matches</h3>
                <ul class="match-list">
        `;

        // Add job matches
        if (Array.isArray(data.matches) && data.matches.length > 0) {
            data.matches.forEach((match, index) => {
                const score = parseFloat(match.score) || 0;
                const percentage = (score * 100).toFixed(1);

                html += `
                    <li class="match-item">
                        <div class="match-header">
                            <span class="match-rank">#${index + 1}</span>
                            <span class="match-title">${escapeHtml(match.title)}</span>
                        </div>
                        <div class="match-score">Similarity: ${match.score} (${percentage}%)</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${percentage}%"></div>
                        </div>
                    </li>
                `;
            });
        } else {
            html += '<li class="match-item">No matches found.</li>';
        }

        html += `
                </ul>
            </div>
        `;

        DOM.result.innerHTML = html;
        DOM.result.style.display = 'block';
        scrollToElement(DOM.result);
    }

    /**
     * Display JD match results
     */
    function displayJDMatch(data) {
        if (!data || typeof data.match_percentage !== 'number') {
            showError('Invalid response from server. Please try again.');
            return;
        }

        const percentage = data.match_percentage;
        const components = data.component_scores || {};
        const missingKeywords = data.missing_keywords || {};
        const keywordSuggestions = data.keyword_suggestions || [];
        const skillsBreakdown = data.skills_breakdown || {};

        let matchLevel = '';
        let matchClass = '';

        // Determine match level
        if (percentage >= 80) {
            matchLevel = 'Excellent Match!';
            matchClass = 'excellent';
        } else if (percentage >= 60) {
            matchLevel = 'Good Match!';
            matchClass = 'good';
        } else if (percentage >= 40) {
            matchLevel = 'Moderate Match';
            matchClass = 'moderate';
        } else {
            matchLevel = 'Low Match';
            matchClass = 'low';
        }

        let html = `
            <h2> JD Match Analysis</h2>
            <div class="match-card ${matchClass}">
                <div class="match-percentage-large">${percentage}%</div>
                <div class="match-level">${matchLevel}</div>
                <div class="match-description">${escapeHtml(data.message || '')}</div>
                <div class="match-bar-container">
                    <div class="match-bar">
                        <div class="match-bar-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            </div>
        `;

        // Add missing keywords section
        if (missingKeywords.critical || missingKeywords.important || missingKeywords.optional) {
            html += `
                <div class="result-card">
                    <h3> Missing Keywords</h3>
                    <p class="section-description">Add these keywords to improve your match score:</p>
            `;

            if (missingKeywords.critical && missingKeywords.critical.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title critical"> Critical (High Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.critical.map(kw => `
                                <span class="keyword-badge critical">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.important && missingKeywords.important.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title important"> Important (Medium Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.important.map(kw => `
                                <span class="keyword-badge important">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            if (missingKeywords.optional && missingKeywords.optional.length > 0) {
                html += `
                    <div class="keywords-section">
                        <h4 class="keywords-title optional"> Optional (Low Priority)</h4>
                        <div class="keywords-list">
                            ${missingKeywords.optional.map(kw => `
                                <span class="keyword-badge optional">${escapeHtml(kw)}</span>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // Add keyword suggestions
            if (keywordSuggestions.length > 0) {
                html += `
                    <div class="suggestions-box">
                        <h4> Recommendations:</h4>
                        <ul class="suggestions-list">
                            ${keywordSuggestions.map(suggestion => `
                                <li>${escapeHtml(suggestion)}</li>
                            `).join('')}
                        </ul>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Add skills breakdown section
        if (skillsBreakdown.missing_skills || skillsBreakdown.matched_skills) {
            html += `
                <div class="result-card">
                    <h3>⚡ Skills Analysis</h3>
            `;

            // Matched skills
            if (skillsBreakdown.matched_skills && Object.keys(skillsBreakdown.matched_skills).length > 0) {
                html += `
                    <div class="skills-section matched">
                        <h4>✅ Matched Skills</h4>
                        <div class="skills-categories">
                `;

                for (const [category, skills] of Object.entries(skillsBreakdown.matched_skills)) {
                    if (skills && skills.length > 0) {
                        html += `
                            <div class="skill-category">
                                <div class="category-name">${formatCategoryName(category)}</div>
                                <div class="skills-list">
                                    ${skills.map(skill => `
                                        <span class="skill-badge matched">${escapeHtml(skill)}</span>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                }

                html += `
                        </div>
                    </div>
                `;
            }

            // Missing skills
            if (skillsBreakdown.missing_skills && Object.keys(skillsBreakdown.missing_skills).length > 0) {
                html += `
                    <div class="skills-section missing">
                        <h4>❌ Missing Skills</h4>
                        <p class="section-description">Consider adding these skills to your resume:</p>
                        <div class="skills-categories">
                `;

                for (const [category, skills] of Object.entries(skillsBreakdown.missing_skills)) {
                    if (skills && skills.length > 0) {
                        html += `
                            <div class="skill-category">
                                <div class="category-name">${formatCategoryName(category)}</div>
                                <div class="skills-list">
                                    ${skills.map(skill => `
                                        <span class="skill-badge missing">${escapeHtml(skill)}</span>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                }

                html += `
                        </div>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Add component breakdown if available
        if (Object.keys(components).length > 0) {
            html += `
                <div class="result-card">
                    <h3> Detailed Score Breakdown</h3>
                    <div class="component-scores">
            `;

            const componentLabels = {
                'semantic': ' Semantic Similarity',
                'keyword': ' Keyword Match',
                'skills': '⚡ Skills Match',
                'context': ' Contextual Match'
            };

            for (const [key, value] of Object.entries(components)) {
                const label = componentLabels[key] || key;
                html += `
                    <div class="component-item">
                        <div class="component-header">
                            <span class="component-label">${label}</span>
                            <span class="component-value">${value}%</span>
                        </div>
                        <div class="component-bar">
                            <div class="component-bar-fill" style="width: ${value}%"></div>
                        </div>
                    </div>
                `;
            }

            html += `
                    </div>
                    <p class="breakdown-note">
                        <small>The final score is calculated using weighted average: 
                        Semantic (40%) + Keywords (30%) + Skills (20%) + Context (10%)</small>
                    </p>
                </div>
            `;
        }

        DOM.jdMatchResult.innerHTML = html;
        DOM.jdMatchResult.style.display = 'block';
        scrollToElement(DOM.jdMatchResult);
    }

    // ==================== UTILITY FUNCTIONS ====================
    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    /**
     * Format file size for display
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * Get education level label
     */
    function getEducationLabel(level) {
        const labels = {
            0: 'Unknown',
            1: "Bachelor's",
            2: "Master's",
            3: 'PhD/Doctorate'
        };
        return labels[level] || 'Unknown';
    }

    /**
     * Get seniority level label
     */
    function getSeniorityLabel(level) {
        const labels = {
            0: 'Entry Level',
            1: 'Mid Level',
            2: 'Senior Level',
            3: 'Lead/Principal'
        };
        return labels[level] || 'Mid Level';
    }

    /**
     * Format category name for display
     */
    function formatCategoryName(category) {
        const names = {
            'programming_languages': ' Programming Languages',
            'web_frameworks': ' Web Frameworks',
            'databases': '️ Databases',
            'cloud_platforms': '☁️ Cloud Platforms',
            'devops_tools': ' DevOps Tools',
            'data_science_ml': ' Data Science & ML',
            'mobile_development': ' Mobile Development',
            'testing_frameworks': ' Testing Frameworks',
            'other_technologies': '⚙️ Other Technologies',
            'methodologies': ' Methodologies',
            'soft_skills': ' Soft Skills',
            'design_tools': ' Design Tools',
            'other_tools': '️ Other Tools'
        };
        return names[category] || category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Scroll to element smoothly
     */
    function scrollToElement(element) {
        setTimeout(() => {
            element.scrollIntoView(CONFIG.SCROLL_OPTIONS);
        }, 100);
    }

    // ==================== START APPLICATION ====================
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();