from typing import Dict, Optional


def build_reflection_rules(reflection: Optional[dict], has_failed_paths: bool) -> str:
    if not isinstance(reflection, dict):
        return (
            'If last outcome = accept:\n'
            '   -> Continue with similar direction\n'
            '   -> Move closer to target directly'
        )

    outcome = reflection.get('outcome', 'none')
    reason = reflection.get('reason_guess', 'none')
    rules = []
    if outcome == 'reject':
        rules.append('If last outcome = reject:\n')
        if reason == 'low_similarity':
            rules.append('1. If reason = low_similarity:')
            rules.append('   -> Choose items MORE SIMILAR to user history')
            rules.append('   -> Avoid low-similarity jumps')
        elif reason == 'wrong_topic':
            rules.append('2. If reason = wrong_topic:')
            rules.append('   -> CHANGE semantic direction')
            rules.append('   -> Do NOT stay in the same topic cluster')
        elif reason == 'too_far':
            rules.append('2. If reason = too_far:')
            rules.append('   -> Increase target alignment')
            rules.append('   -> Avoid bridge items with weak acceptance probability')
        if has_failed_paths:
            rules.append('3. If the same path structure failed before:')
            rules.append('   -> DO NOT repeat similar paths')
    elif outcome == 'accept':
        rules.append('If last outcome = accept:')
        rules.append('   -> Continue with similar direction')
        rules.append('   -> Move closer to target directly')
    else:
        rules.append('If last outcome = accept:')
        rules.append('   -> Continue with similar direction')
        rules.append('   -> Move closer to target directly')
    return '\n'.join(rules)


def build_target_oriented_planner_prompt(
    history_titles: str,
    user_profile: str,
    target_text: str,
    candidate_item_lines: str,
    last_action: str,
    outcome: str,
    failure_reason: str,
    rec_sim: float,
    target_sim: float,
    policy_bias: Dict[str, float],
    reflection: Optional[dict],
    llm_path_k: int,
    reject_streak: int,
    failed_path_lines: str,
    candidate_path_lines: str,
    graph_topology_text: str,
    reflection_triplet_text: str,
    target_id: str,
) -> str:
    reflection_rules = build_reflection_rules(reflection, has_failed_paths=failed_path_lines != 'none')
    return (
        'You are a target-oriented recommendation planner.\n\n'
        'Your goal is NOT to generate a long or interesting path.\n'
        'Your PRIMARY goal is to reach the target item as efficiently as possible.\n'
        'User acceptance is SECONDARY and should only be used to support target-reaching decisions.\n\n'
        'You must prioritize:\n'
        '1. Reach the target in the fewest valid steps\n'
        '2. Strong semantic similarity and monotonic progress toward the target\n'
        '3. Consistency with user history only when it helps the target-reaching plan\n\n'
        'Constraint (VERY IMPORTANT):\n'
        'For a valid path P = [p1, p2, ..., target], the target similarity must increase at every step:\n'
        'cos(p1, target) < cos(p2, target) < ... < cos(target, target).\n'
        'Each step MUST strictly move closer to the target.\n\n'
        'You must AVOID:\n'
        '- Unnecessary long paths\n'
        '- Exploration without clear target-reaching benefit\n'
        '- Bridge items that do not clearly move closer to the target\n'
        '- Repeating previously failed patterns\n'
        '\nIf a longer path does NOT improve target similarity faster, ALWAYS prefer the shorter valid path.\n'
        'The first step should be aligned with high baseline score items.\n\n'
        '[User History]\n'
        f'{history_titles}\n\n'
        '[Target Item]\n'
        f'{target_text}\n\n'
        '[Candidate Items]\n'
        f'{candidate_item_lines}\n\n'
        '[Intent Graph Topology]\n'
        f'{graph_topology_text}\n\n'
        '[Last Interaction]\n'
        f'Action: {last_action}\n'
        f'Outcome: {outcome}\n\n'
        f'Reason: {failure_reason}\n\n'
        '[Reflection Signal]\n'
        f'{reflection_triplet_text}\n\n'
        '[Current Strategy Weights]\n'
        f'target_weight: {policy_bias["target_weight"]:.2f}\n'
        f'history_weight: {policy_bias["history_weight"]:.2f}\n'
        f'bridge_weight: {policy_bias["bridge_weight"]:.2f}\n\n'
        '[Constraints]\n'
        f'- Path length <= {llm_path_k}\n'
        '- Last node MUST be target item\n'
        '- All nodes must be from candidate set\n\n'
        '- Every intermediate step MUST have higher target similarity than the previous step\n'
        '- If two paths reach the target similarly well, prefer the shorter one\n'
        '- The first step should stay close to high baseline-score candidates\n'
        '- Do not add an intermediate item unless it improves target reachability\n\n'
        '[Planning Strategy]\n\n'
        f'reject_streak: {reject_streak}\n\n'
        'If reject_streak >= 2:\n'
        '   -> You may try a DIFFERENT path\n'
        '   -> But still keep the path short and target-progressive\n\n'
        'Otherwise:\n'
        '   -> Choose the SHORTEST path that can reach the target\n\n'
        '[Reflection Rules - MUST FOLLOW]\n\n'
        f'{reflection_rules}\n\n'
        '[Auxiliary Context]\n'
        f'User profile summary: {user_profile}\n'
        'Failure analysis:\n'
        f'- reason: {failure_reason}\n'
        f'- rec_sim: {rec_sim:.4f}\n'
        f'- target_sim: {target_sim:.4f}\n\n'
        '[FAILED PATHS]\n'
        'Previously failed paths:\n'
        f'{failed_path_lines}\n\n'
        '[CANDIDATE PATHS]\n'
        f'{candidate_path_lines}\n\n'
        '[TASK]\n'
        'Select the BEST path that reaches the target as quickly as possible with the minimum necessary detours.\n\n'
        'Reject any path that does not strictly increase target similarity step by step.\n\n'
        'Return JSON only:\n\n'
        '{\n'
        f'  "best_path": ["item1", "item2", ..., "{target_id}"],\n'
        '  "reason": "brief explanation of why this path is the shortest effective target-reaching plan"\n'
        '}'
    )