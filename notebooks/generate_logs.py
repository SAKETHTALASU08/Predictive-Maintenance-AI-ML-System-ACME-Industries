import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define components and failure modes
components = [
    "Drive Belt",
    "Hydraulic Pump",
    "CNC Spindle",
    "Bearing Assembly",
    "Motor Housing"
]

failure_modes = [
    "Tool Wear Failure",          # TWF
    "Heat Dissipation Failure",   # HDF
    "Power Failure",              # PWF
    "Overstrain Failure",         # OSF
    "Random Failure"              # RNF
]

# ID mappings
component2id = {c: i for i, c in enumerate(components)}
failure2id = {f: i for i, f in enumerate(failure_modes)}

# Generate variations dictionary
# Expanded synonyms representing casual language, technical terms, and noisy descriptors
comp_synonyms = {
    "Drive Belt": [
        "drive belt", "belt", "poly-v belt", "v-belt", "tensioner belt", "driv belt", "belts",
        "conveyor belt", "tensioner", "belt area", "conveyor", "belts assembly", "tension pulley"
    ],
    "Hydraulic Pump": [
        "hydraulic pump", "pump", "hydraulic unit", "hpu", "hydraulic line", "hydrualic pump", "HPU unit",
        "hydraulics", "fluid line", "hydraulics area", "pump unit", "pressure gauge", "hydraulics block"
    ],
    "CNC Spindle": [
        "CNC spindle", "spindle", "spindle unit", "spndl", "spinle", "chuck motor", "spindle assembly",
        "cnc head", "spindle head", "chuck", "tool head", "spindle motor", "spindle casing"
    ],
    "Bearing Assembly": [
        "bearing assembly", "bearing", "bearings", "bearing kit", "bearign", "assembly bearings",
        "bearing block", "shaft bearing", "bearings area", "ball bearings", "bearing housing"
    ],
    "Motor Housing": [
        "motor housing", "housing", "casing", "motor casing", "housign", "housing shield",
        "outer shell", "casing shield", "motor area", "back unit", "back casing", "housing unit"
    ]
}

# Symptom templates for each failure mode with added noise and messy operator phrases
symptom_templates = {
    "Tool Wear Failure": [
        "is completely worn out, tool wear value is critical",
        "tool wear ratio exceeded limits, tool is extremely dull",
        "experiencing severe pitting, tool wear min reading is at maximum",
        "wear value too high, tool wear min reached threshold",
        "tool is dull and worn, needs replacement immediately",
        "warn out tool, wear min is at critical levels",
        "severe tool wear detected on inspection",
        "tool wear min is maxed out, cutting surface is damaged",
        "dull blade / tool wear critical",
        "tool wear min critical, pitting visible on the contact point",
        "tool worn down to the nub, replace now",
        "high tool wear detected during cycle check",
        "tool wear is at redline, quality is dropping",
        "worn tool causing rough finish on workpiece",
        "tool wear value is 240min, way past replacement point",
        "dull cutting tool, wear min exceeded safety margin",
        "excessive wear on tool head",
        "tool wears out too fast, current wear min is critical",
        "replace tool, wear min value is red",
        "worn tool bit, replacing now before failure",
        "worn out", "looks worn", "dull", "rough finish", "pitting",
        "slipping and worn", "blade is dull", "tool is worn out completely"
    ],
    "Heat Dissipation Failure": [
        "is overheating rapidly, thermal camera shows 95C",
        "temperature spike detected, cooling fan seems broken",
        "is hot to the touch, smoke coming out of housing",
        "thermal runaway warning, temp delta is critical",
        "heating up fast, temperature is at critical 105C",
        "smoke coming from unit, cooling failed",
        "severe overheating, ventilation duct is blocked",
        "temperature is at 98 degrees C, thermal sensor alert",
        "overheating alert, cooling paste degradation suspected",
        "temp is way too hot, casing is smoking",
        "hot casing, smoke visible, cooling system down",
        "thermal paste dry, unit overheating under load",
        "abnormal heat buildup, coolant level low",
        "temperature spiked to critical limits, fan stopped",
        "unit running extremely hot, potential fire hazard",
        "thermal overload tripped due to high heat",
        "ventilation shaft blocked, temperature rising",
        "smells like burning plastic, unit is overheating",
        "temp sensor reading red, cooling failure imminent",
        "casing temp too high, thermal cutoff engaged",
        "overheating", "hot", "smoking", "too hot to touch", "thermal overload",
        "hot and shaking", "weird smell", "smoking slightly", "abnormal heat"
    ],
    "Power Failure": [
        "lost power completely, voltage dropped to zero",
        "power failure, electrical breaker tripped in the cabinet",
        "fuse blown, voltage surge detected on line",
        "no power supply, transformer failure suspected",
        "sudden voltage drop, machine stopped mid-cycle",
        "electrical fault, voltage supply lines went dead",
        "breaker tripped, power failure warning on control panel",
        "voltage surge tripped the breaker, no power to the HPU",
        "blown fuse in electrical panel, power cutoff",
        "lost voltage, mains power failed during tool run",
        "mains power failure, backup generator running",
        "blown power relay, voltage went to 0V",
        "power fluctuation, voltage drop triggered safety cutoff",
        "breaker trips instantly when turned on, short circuit suspected",
        "electrical arc damage on wiring, power cut off",
        "no voltage reading at the terminals",
        "fuse pop, power supply lines went down",
        "electrical breaker failed, power cut to the spindle",
        "voltage drop under load, power supply faulty",
        "complete blackout in cabinet, power fail",
        "no power", "breaker tripped", "fuse popped", "voltage dropped", "power out",
        "lost power", "electrical fault", "breaker trips"
    ],
    "Overstrain Failure": [
        "is completely seized and jammed, motor stalled under load",
        "stuck and seized up, overstrain torque limit exceeded",
        "is jammed, high torque reading triggered emergency stop",
        "stalled under heavy load, overstrain torque overload",
        "cannot rotate, drive shaft is seized",
        "stuck due to high friction, torque limit reached",
        "torque reading exceeded 85 Nm, mechanical jam detected",
        "stalled mid-operation, high friction torque overload",
        "jammed drive, motor stalled under stress",
        "severe mechanical stress, spindle seized up",
        "jammed housing, torque spikes to max",
        "stalled motor, torque limits exceeded safety margin",
        "seized shaft, gears jammed",
        "torque overload, mechanical lockup detected",
        "stalled under torque load, mechanical strain",
        "stuck assembly, drive belt snapped due to overstrain",
        "mechanical jam, torque is pegged at max",
        "stalled drive shaft, unit is seized",
        "overstrain torque overload, HPU stalled",
        "gears seized up, complete mechanical stall",
        "seized up", "jammed", "stalled", "stuck", "cannot rotate",
        "cant hold pressure", "grinding noise", "friction overload"
    ],
    "Random Failure": [
        "shut down with unexplained error code ERR-404",
        "random glitch, machine rebooted spontaneously",
        "intermittent communication loss, control panel fault",
        "random shutdown, no error codes in log history",
        "unexplained behavior, machine stops randomly",
        "unidentified glitch, controller went offline",
        "random trip, control interface lost connection",
        "unexplained sensor fault, error code ERR-502",
        "intermittent connection issue, sensor signal dropped",
        "glitchy controls, random error triggered safety loop",
        "controller board rebooted randomly during cycle",
        "random failure code ERR-99, control panel locked",
        "intermittent signal loss from sensors",
        "unexplained halt, system needs a hard reboot",
        "random error, control loop failed",
        "unidentified fault, machine halted without alarm details",
        "glitch in sensor data transmission, system reset",
        "unexplained shutdown during start sequence",
        "intermittent connection faults, CAN bus error",
        "random emergency loop trip, cause unknown",
        "broke again", "wierd noise", "keeps stopping", "stopped", "leaking",
        "oil everywhere", "weird noise", "stops randomly", "glitchy connection"
    ]
}

# Operator language patterns with diverse noise profiles (casual, typo-laden, terse, vague)
# Note: Every pattern must contain {symptom} or {symptom_short} to ensure uniqueness per failure mode.
operator_patterns = [
    "Inspection reveals that the {comp} {symptom}.",
    "The {comp} {symptom}. Operations are affected, check ASAP.",
    "the {comp_typo} is {symptom} again. please check.",
    "{comp} {symptom_short}.",
    "Warning: {comp} {symptom} and smells weird.",
    "Operator reports: '{comp} {symptom_colloquial}.'",
    "Maintenance log: {comp} {symptom}.",
    "operator says {comp_typo} {symptom_typo} today.",
    "CRITICAL: {comp} {symptom_short} immediately!",
    "Standard check: {comp} {symptom} during run.",
    "thing near {comp} {symptom}.",
    "{comp} making wierd noise, feels like {symptom_short}.",
    "somthing grinding in the {comp_casual} area, check for {symptom_short}.",
    "its hot and shaking alot near the {comp}, check for {symptom_short}.",
    "weird smell from the {comp_casual} unit, possibly {symptom_short}.",
    "keeps stopping every few minutes, {comp} is {symptom_short}.",
    "oil everywhere near the {comp_casual}, suspect {symptom_short}.",
    "{comp} cant hold pressure, seems like {symptom_short}.",
    "vibrating way more than normal near the {comp_casual}, {symptom_short}.",
    "leaking or snapped near the {comp_casual}, {symptom_short} suspected."
]

def make_typo(text):
    typos = {
        "belt": "blt",
        "drive": "driv",
        "hydraulic": "hydrualic",
        "pump": "pmp",
        "spindle": "spinle",
        "bearing": "bearign",
        "housing": "housign",
        "worn": "warn",
        "critical": "critcal",
        "temperature": "tempurature",
        "overheating": "overheating",
        "voltage": "voltag",
        "seized": "siezed",
        "jammed": "jamed",
        "random": "randm",
        "failure": "failur",
        "machine": "mchine",
        "weird": "wierd",
        "something": "somthing"
    }
    words = text.split()
    new_words = []
    for w in words:
        wl = w.lower().strip(".,'\"")
        if wl in typos:
            new_word = w.replace(wl, typos[wl])
            new_words.append(new_word)
        else:
            if len(w) > 4 and np.random.rand() < 0.15:
                idx = np.random.randint(1, len(w)-2)
                w_list = list(w)
                w_list[idx], w_list[idx+1] = w_list[idx+1], w_list[idx]
                new_words.append("".join(w_list))
            else:
                new_words.append(w)
    return " ".join(new_words)

# Generate dataset
generated_data = []
np.random.seed(42)

# We generate exactly 40 variations per class combination (5 * 5 * 40 = 1000 samples)
for comp in components:
    for fail in failure_modes:
        comp_id = component2id[comp]
        fail_id = failure2id[fail]
        
        syns = comp_synonyms[comp]
        syms = symptom_templates[fail]
        
        for v in range(40):
            comp_syn = syns[v % len(syns)]
            symptom_base = syms[v % len(syms)]
            
            pattern_idx = v % len(operator_patterns)
            
            comp_val = comp_syn
            symptom_val = symptom_base
            comp_typo = make_typo(comp_val)
            symptom_typo = make_typo(symptom_val)
            comp_casual = comp_val
            
            # Extract a shorter symptom phrase
            symptom_words = symptom_val.split()
            symptom_short = " ".join(symptom_words[:3]) if len(symptom_words) > 3 else symptom_val
            symptom_colloquial = symptom_val
            
            pattern = operator_patterns[pattern_idx]
            text = pattern.format(
                comp=comp_val,
                symptom=symptom_val,
                comp_typo=comp_typo,
                symptom_typo=symptom_typo,
                comp_casual=comp_casual,
                symptom_short=symptom_short,
                symptom_colloquial=symptom_colloquial
            )
            
            generated_data.append({
                "log_text": text,
                "component": comp,
                "failure_mode": fail,
                "component_id": comp_id,
                "failure_mode_id": fail_id
            })

df_logs = pd.DataFrame(generated_data)

# Print stats
print(f"Generated total logs: {len(df_logs)}")
print(df_logs["component"].value_counts())
print(df_logs["failure_mode"].value_counts())

# Ensure directory exists
os.makedirs("data", exist_ok=True)
df_logs.to_csv("data/maintenance_logs.csv", index=False)
print("Saved data/maintenance_logs.csv")

# ─── Stratified Split ────────────────────────────────────────────────────────
# Combine component and failure_mode to make a stratification column
df_logs["strat_class"] = df_logs["component"] + "_" + df_logs["failure_mode"]

# First split: 80% train+val, 20% test
df_temp, df_test = train_test_split(
    df_logs, test_size=0.20, random_state=42, stratify=df_logs["strat_class"]
)

# Second split: 60% train, 20% val from the remaining 80% (which is 0.25 of temp)
df_train, df_val = train_test_split(
    df_temp, test_size=0.25, random_state=42, stratify=df_temp["strat_class"]
)

# Drop helper column
df_train = df_train.drop(columns=["strat_class"])
df_val = df_val.drop(columns=["strat_class"])
df_test = df_test.drop(columns=["strat_class"])

print("\nSplit Sizes:")
print(f"  Training   : {len(df_train)} rows")
print(f"  Validation : {len(df_val)} rows")
print(f"  Test       : {len(df_test)} rows")

# Save to csv files
df_train.to_csv("data/train_logs.csv", index=False)
df_val.to_csv("data/val_logs.csv", index=False)
df_test.to_csv("data/test_logs.csv", index=False)
print("Saved splits under data/ directory.")
