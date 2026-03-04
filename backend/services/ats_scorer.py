"""
ATS Scorer — Main Scoring Class

Provides ATSScorer with dual analysis modes (quick + deep) and
context-aware suggestion generation. All constants live in
ats_constants.py; all helper/detection functions live in ats_helpers.py.
"""

import re
from typing import Dict, List, Any

from backend.services.ats_constants import (
    SCORING_WEIGHTS,
    SECTION_KEYWORD_WEIGHT,
    FREQUENCY_BONUS_PER_EXTRA,
    FREQUENCY_BONUS_MAX_MENTIONS,
    FREQUENCY_BONUS_MAX_TOTAL,
)
from backend.services.ats_helpers import (
    detect_resume_sections,
    extract_experience_duration,
    detect_education_level,
    detect_seniority_level,
    extract_achievements,
    classify_jd_keywords,
    match_job_title,
    calculate_formatting_penalty,
    detect_keyword_stuffing,
    calculate_recency_bonus,
    get_field_weight_for_keyword,
    get_score_interpretation,
)


# ==================== SUGGESTION GENERATOR ====================

def generate_suggestions(
    missing_keywords: Dict[str, List[str]],
    achievements: List[Dict],
    experience: Dict,
    sections: Dict[str, str],
    formatting_penalty: Dict,
    skill_score: int = 0,
    title_score: int = 0,
    education_score: int = 0,
    matched_skills: Dict[str, List[str]] = None
) -> List[str]:
    """
    Generate dynamic, context-aware improvement suggestions.

    Suggestions are prioritized and specific to the actual gaps found.
    """
    suggestions = []

    # Priority 1: Critical missing keywords (most impactful)
    critical_missing = missing_keywords.get('critical', [])
    if critical_missing:
        if len(critical_missing) >= 5:
            top_3 = critical_missing[:3]
            suggestions.append(
                f"CRITICAL: Add these required skills to your resume: {', '.join(top_3)} "
                f"(+{len(critical_missing) - 3} more missing)"
            )
        elif len(critical_missing) >= 2:
            suggestions.append(
                f"Add these required keywords: {', '.join(critical_missing)}"
            )
        else:
            suggestions.append(
                f"Add the required keyword '{critical_missing[0]}' to match the job description"
            )

    # Priority 2: Important missing keywords
    important_missing = missing_keywords.get('important', [])
    if important_missing and len(suggestions) < 4:
        top_important = important_missing[:3]
        suggestions.append(
            f"Consider adding preferred skills: {', '.join(top_important)}"
        )

    # Priority 3: Skill match suggestions (based on actual score)
    if skill_score is not None:
        if skill_score < 30:
            suggestions.append(
                "Your skill match is low. Review the job description and add relevant technologies you know"
            )
        elif skill_score < 50:
            if matched_skills:
                matched_count = sum(len(v) for v in matched_skills.values())
                if matched_count < 5:
                    suggestions.append(
                        f"Only {matched_count} skills matched. List more of your technical skills explicitly"
                    )

    # Priority 4: Achievement suggestions (dynamic based on count)
    achievement_count = len(achievements)
    if achievement_count == 0:
        suggestions.append(
            "Add quantified achievements with metrics (%, $, numbers) to show impact"
        )
    elif achievement_count < 3:
        needed = 3 - achievement_count
        suggestions.append(
            f"Add {needed} more quantified achievements. Examples: 'Increased X by 25%', 'Reduced Y by $10K'"
        )

    # Priority 5: Experience suggestions (based on actual data)
    total_years = experience.get('total_years', 0)
    skill_years = experience.get('skill_years', {})

    if total_years == 0:
        suggestions.append(
            "Specify your total years of experience clearly (e.g., '5+ years of experience')"
        )
    elif not skill_years:
        suggestions.append(
            "Add years of experience for key skills (e.g., 'Python - 4 years', 'AWS - 2 years')"
        )

    # Priority 6: Section suggestions (specific to what's missing)
    missing_sections = []
    if 'skills' not in sections:
        missing_sections.append('Skills')
    if 'summary' not in sections:
        missing_sections.append('Professional Summary')
    if 'projects' not in sections and 'experience' not in sections:
        missing_sections.append('Projects or Experience')

    if missing_sections and len(suggestions) < 5:
        suggestions.append(
            f"Add missing sections: {', '.join(missing_sections)}"
        )

    # Priority 7: Title match suggestions
    if title_score is not None and title_score < 50:
        suggestions.append(
            "Include the target job title or related titles in your resume summary"
        )

    # Priority 8: Education suggestions
    if education_score is not None and education_score < 40:
        suggestions.append(
            "Clearly list your educational qualifications with degree names"
        )

    # Priority 9: Formatting suggestions (specific)
    for detail in formatting_penalty.get('details', []):
        if 'bullet' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Use more bullet points (aim for 3-5 per job) to highlight key accomplishments"
            )
        elif 'contact' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Add complete contact information: email, phone, LinkedIn profile"
            )
        elif 'date' in detail.lower() and len(suggestions) < 6:
            suggestions.append(
                "Use specific date ranges (e.g., 'Jan 2020 - Dec 2023') instead of vague dates"
            )

    # Priority 10: General improvement if score is still low
    if len(suggestions) < 3:
        optional_missing = missing_keywords.get('optional', [])
        if optional_missing and len(optional_missing) > 3:
            suggestions.append(
                f"Bonus: Include optional skills like {', '.join(optional_missing[:2])} if applicable"
            )

    # Return top 6 most impactful suggestions
    return suggestions[:6]


# ==================== MAIN SCORER CLASS ====================

class ATSScorer:
    """
    ATS Score Calculator with configurable weights and dual analysis modes.
    """

    def __init__(self, mode: str = "deep", weights: Dict[str, float] = None):
        """
        Initialize ATS Scorer.

        Args:
            mode: "quick" for fast keyword matching, "deep" for full analysis
            weights: Custom scoring weights (optional)
        """
        self.mode = mode
        self.weights = weights or SCORING_WEIGHTS.copy()

    def calculate_skill_score(
        self,
        resume_skills: List[str],
        jd_keywords: Dict[str, List[str]],
        semantic_similarity: float = None,
        sections: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Calculate skill match score with field-weighting.

        Keywords found in Experience/Projects sections are weighted higher
        than those in a plain Skills list.
        """
        resume_skills_lower = [s.lower() for s in resume_skills]

        matched = {
            'required': [],
            'preferred': [],
            'standard': []
        }
        missing = {
            'critical': [],
            'important': [],
            'optional': []
        }
        field_weights_sum = 0.0
        field_weights_max = 0.0

        # Check required keywords (weight: 2x)
        for keyword in jd_keywords.get('required', []):
            if keyword.lower() in resume_skills_lower:
                matched['required'].append(keyword)
                if sections:
                    field_weights_sum += get_field_weight_for_keyword(keyword, sections) * 2
                else:
                    field_weights_sum += 2.0
            else:
                missing['critical'].append(keyword)
            field_weights_max += 2.0 * SECTION_KEYWORD_WEIGHT.get('experience', 1.5)

        # Check preferred keywords (weight: 1x)
        for keyword in jd_keywords.get('preferred', []):
            if keyword.lower() in resume_skills_lower:
                matched['preferred'].append(keyword)
                if sections:
                    field_weights_sum += get_field_weight_for_keyword(keyword, sections)
                else:
                    field_weights_sum += 1.0
            else:
                missing['important'].append(keyword)
            field_weights_max += SECTION_KEYWORD_WEIGHT.get('experience', 1.5)

        # Check standard keywords
        for keyword in jd_keywords.get('standard', []):
            if keyword.lower() in resume_skills_lower:
                matched['standard'].append(keyword)
                if sections:
                    field_weights_sum += get_field_weight_for_keyword(keyword, sections) * 0.5
                else:
                    field_weights_sum += 0.5
            else:
                missing['optional'].append(keyword)
            field_weights_max += 0.5 * SECTION_KEYWORD_WEIGHT.get('experience', 1.5)

        # Field-weighted score (0-100)
        if field_weights_max > 0:
            base_score = (field_weights_sum / field_weights_max) * 100
        else:
            total = len(jd_keywords.get('required', [])) + len(jd_keywords.get('preferred', [])) + len(jd_keywords.get('standard', []))
            matched_count = len(matched['required']) + len(matched['preferred']) + len(matched['standard'])
            base_score = (matched_count / max(total, 1)) * 100

        # Add semantic similarity boost in deep mode
        if self.mode == "deep" and semantic_similarity is not None:
            semantic_boost = semantic_similarity * 10
            base_score = min(100, base_score + semantic_boost)

        # Keyword frequency bonus: keywords mentioned 2-3× show
        # natural, contextual usage (proof in multiple places).
        if self.mode == "deep" and resume_skills:
            frequency_bonus = 0.0
            all_matched = (
                matched['required'] + matched['preferred'] + matched['standard']
            )
            for kw in all_matched:
                count = resume_skills_lower.count(kw.lower())
                if count < 2 and sections:
                    section_hits = sum(
                        1 for sec in sections.values()
                        if kw.lower() in sec.lower()
                    )
                    count = max(count, section_hits)
                extra = min(count - 1, FREQUENCY_BONUS_MAX_MENTIONS - 1)
                if extra > 0:
                    frequency_bonus += extra * FREQUENCY_BONUS_PER_EXTRA
            frequency_bonus = min(frequency_bonus, FREQUENCY_BONUS_MAX_TOTAL)
            base_score = min(100, base_score + frequency_bonus)

        return {
            'score': int(base_score),
            'matched': matched,
            'missing': missing
        }

    def calculate_title_score(self, resume_text: str, jd_title: str) -> Dict[str, Any]:
        """Calculate job title match score."""
        title_match = match_job_title(resume_text, jd_title)
        seniority = detect_seniority_level(resume_text)

        base_score = title_match['score']
        seniority_bonus = seniority['level_score'] * 5

        return {
            'score': min(100, base_score + seniority_bonus),
            'title_match': title_match,
            'seniority': seniority
        }

    def calculate_experience_score(
        self,
        resume_experience: Dict,
        required_years: int = 0
    ) -> Dict[str, Any]:
        """Calculate experience score."""
        total_years = resume_experience.get('total_years', 0)

        if required_years > 0:
            ratio = total_years / required_years
            score = min(100, int(ratio * 100))
        else:
            if total_years >= 10:
                score = 100
            elif total_years >= 5:
                score = 80
            elif total_years >= 3:
                score = 60
            elif total_years >= 1:
                score = 40
            else:
                score = 20

        return {
            'score': score,
            'total_years': total_years,
            'skill_years': resume_experience.get('skill_years', {})
        }

    def calculate_achievement_score(self, achievements: List[Dict]) -> Dict[str, Any]:
        """Calculate achievement score based on quantified accomplishments."""
        count = len(achievements)

        if count >= 5:
            score = 100
        elif count >= 3:
            score = 75
        elif count >= 1:
            score = 50
        else:
            score = 20

        return {
            'score': score,
            'count': count,
            'achievements': achievements[:5]
        }

    def calculate_education_score(
        self,
        resume_education: Dict,
        required_level: str = None
    ) -> Dict[str, Any]:
        """Calculate education score."""
        level = resume_education.get('highest_level', 'unknown')
        level_score = resume_education.get('level_score', 0)

        if required_level:
            required_score = {'phd': 4, 'masters': 3, 'bachelors': 2, 'diploma': 1}.get(required_level.lower(), 0)
            if level_score >= required_score:
                score = 100
            elif level_score == required_score - 1:
                score = 70
            else:
                score = 40
        else:
            score = level_score * 25

        return {
            'score': max(score, 20),
            'level': level,
            'degrees_found': resume_education.get('degrees_found', [])
        }

    def calculate_ats_score(
        self,
        resume_text: str,
        jd_text: str,
        jd_title: str = "",
        required_years: int = 0,
        required_education: str = None,
        resume_skills: List[str] = None,
        jd_keywords: List[str] = None,
        semantic_similarity: float = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive ATS score.

        Args:
            resume_text: Full resume text
            jd_text: Job description text
            jd_title: Job title (optional)
            required_years: Required years of experience (optional)
            required_education: Required education level (optional)
            resume_skills: Pre-extracted resume skills (optional)
            jd_keywords: Pre-extracted JD keywords (optional)
            semantic_similarity: Pre-computed semantic similarity (optional)

        Returns:
            Comprehensive score breakdown with suggestions
        """
        if self.mode == "quick":
            return self._quick_scan(resume_text, jd_text, resume_skills, jd_keywords)

        return self._deep_analysis(
            resume_text, jd_text, jd_title,
            required_years, required_education,
            resume_skills, jd_keywords, semantic_similarity
        )

    def _quick_scan(
        self,
        resume_text: str,
        jd_text: str,
        resume_skills: List[str] = None,
        jd_keywords: List[str] = None
    ) -> Dict[str, Any]:
        """Quick keyword-matching scan."""
        if not resume_skills:
            resume_skills = re.findall(r'\b\w+\b', resume_text.lower())
        if not jd_keywords:
            jd_keywords = re.findall(r'\b\w+\b', jd_text.lower())

        resume_set = set(resume_skills)
        jd_set = set(jd_keywords)

        matched = list(resume_set & jd_set)
        missing = list(jd_set - resume_set)

        match_ratio = len(matched) / max(len(jd_set), 1)
        score = int(match_ratio * 100)

        return {
            'ats_score': score,
            'mode': 'quick',
            'interpretation': get_score_interpretation(score),
            'matched_keywords': matched[:20],
            'missing_keywords': {'critical': missing[:10], 'important': [], 'optional': []},
            'suggestions': ['Run Deep Analysis for detailed breakdown']
        }

    def _deep_analysis(
        self,
        resume_text: str,
        jd_text: str,
        jd_title: str,
        required_years: int,
        required_education: str,
        resume_skills: List[str],
        jd_keywords: List[str],
        semantic_similarity: float
    ) -> Dict[str, Any]:
        """Full deep analysis with all sub-scores."""

        # Detect resume sections
        sections = detect_resume_sections(resume_text)

        # Extract experience
        experience = extract_experience_duration(resume_text)

        # Detect education
        education = detect_education_level(resume_text)

        # Extract achievements
        achievements = extract_achievements(resume_text)

        # Classify JD keywords
        if jd_keywords:
            classified_keywords = classify_jd_keywords(jd_text, jd_keywords)
        else:
            classified_keywords = {'required': [], 'preferred': [], 'standard': []}

        # Calculate formatting penalty
        formatting = calculate_formatting_penalty(resume_text, sections)

        # Keyword stuffing detection
        all_kws = (
            classified_keywords.get('required', []) +
            classified_keywords.get('preferred', []) +
            classified_keywords.get('standard', [])
        )
        stuffing = detect_keyword_stuffing(resume_text, all_kws)

        # Recency bonus
        recency = calculate_recency_bonus(resume_text)

        # Calculate sub-scores (with section-aware field-weighting)
        skill_result = self.calculate_skill_score(
            resume_skills or [], classified_keywords, semantic_similarity,
            sections=sections
        )

        title_result = self.calculate_title_score(resume_text, jd_title)

        experience_result = self.calculate_experience_score(experience, required_years)

        achievement_result = self.calculate_achievement_score(achievements)

        education_result = self.calculate_education_score(education, required_education)

        # Calculate final weighted score
        final_score = (
            self.weights['skill_match'] * skill_result['score'] +
            self.weights['title_match'] * title_result['score'] +
            self.weights['experience'] * experience_result['score'] +
            self.weights['achievement'] * achievement_result['score'] +
            self.weights['education'] * education_result['score'] -
            self.weights['formatting_penalty'] * formatting['total']
        )

        # Apply recency bonus/penalty
        final_score += recency['bonus']

        # Apply stuffing penalty
        if stuffing['is_stuffed']:
            final_score -= stuffing['penalty']

        # Required keyword hard gate
        total_required = len(classified_keywords.get('required', []))
        missing_required = len(skill_result['missing'].get('critical', []))
        if total_required > 0 and missing_required > (total_required / 2):
            final_score = min(final_score, 50)

        final_score = max(0, min(100, int(final_score)))

        # Generate dynamic suggestions with context
        suggestions = generate_suggestions(
            skill_result['missing'],
            achievements,
            experience,
            sections,
            formatting,
            skill_score=skill_result['score'],
            title_score=title_result['score'],
            education_score=education_result['score'],
            matched_skills=skill_result['matched']
        )

        # Add stuffing warning if detected
        if stuffing['is_stuffed']:
            suggestions.insert(0,
                f"WARNING: Keyword stuffing detected ({', '.join(stuffing['flagged_keywords'][:3])}). "
                f"Real ATS systems penalize excessive repetition."
            )

        # Add recency suggestion if stale
        if recency['bonus'] < 0:
            suggestions.append(
                f"Your most recent experience is from {recency['most_recent_year']}. "
                f"Update with current or recent roles/projects."
            )

        # Build matched keywords with importance
        matched_keywords = []
        for kw in skill_result['matched'].get('required', []):
            matched_keywords.append({'keyword': kw, 'importance': 'required'})
        for kw in skill_result['matched'].get('preferred', []):
            matched_keywords.append({'keyword': kw, 'importance': 'preferred'})
        for kw in skill_result['matched'].get('standard', [])[:10]:
            matched_keywords.append({'keyword': kw, 'importance': 'standard'})

        return {
            'ats_score': final_score,
            'mode': 'deep',
            'interpretation': get_score_interpretation(final_score),
            'sub_scores': {
                'skill_match': skill_result['score'],
                'title_match': title_result['score'],
                'experience': experience_result['score'],
                'achievement': achievement_result['score'],
                'education': education_result['score'],
                'formatting_penalty': formatting['total'],
                'recency_bonus': recency['bonus'],
                'stuffing_penalty': stuffing['penalty'] if stuffing['is_stuffed'] else 0,
            },
            'matched_keywords': matched_keywords,
            'missing_keywords': skill_result['missing'],
            'achievements_found': [a['text'] for a in achievement_result['achievements']],
            'experience_details': {
                'total_years': experience_result['total_years'],
                'skill_years': experience_result['skill_years'],
                'recency': recency['details'],
            },
            'education_details': {
                'level': education_result['level'],
                'degrees': education_result['degrees_found']
            },
            'sections_detected': list(sections.keys()),
            'stuffing_detected': stuffing['is_stuffed'],
            'required_gate_applied': (
                total_required > 0 and missing_required > (total_required / 2)
            ),
            'suggestions': suggestions
        }


# Convenience function for external use
def calculate_ats_score(
    resume_text: str,
    jd_text: str,
    mode: str = "deep",
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate ATS score for resume against job description.

    Args:
        resume_text: Full resume text
        jd_text: Job description text
        mode: "quick" or "deep" analysis
        **kwargs: Additional parameters passed to ATSScorer

    Returns:
        Comprehensive ATS score breakdown
    """
    scorer = ATSScorer(mode=mode)
    return scorer.calculate_ats_score(resume_text, jd_text, **kwargs)
