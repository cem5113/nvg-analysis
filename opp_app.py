import streamlit as st
import pandas as pd
import numpy as np
import io
import ast

st.title("Operator Panel - NVG Evaluation Consolidation")

# === Constants ===
criteria = ["Depth Perception", "Clarity", "Halo Effect", "Adjustment", "Contrast", "Weight"]
alternatives = ["NVG A", "NVG B", "NVG C", "NVG D", "NVG E", "NVG F", "NVG G"]

# === Step 1: Upload Files ===
st.header("Step 1: Upload User Files")

uploaded_files = st.file_uploader("Upload multiple user selection files:", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    pairwise_matrices = []
    alternative_ratings = []
    user_ids = []

    for file in uploaded_files:
        xls = pd.ExcelFile(file)

        # Read user ID
        df_id = pd.read_excel(xls, sheet_name='User Info')
        user_ids.append(df_id['User ID'][0])

        # Read pairwise matrix
        df_matrix = pd.read_excel(xls, sheet_name='Criteria Comparison', index_col=0)
        pairwise_matrices.append(df_matrix.values)

        # Read alternative ratings
        df_ratings = pd.read_excel(xls, sheet_name='Alternative Ratings', index_col=0)
        alternative_ratings.append(df_ratings)

    st.success(f"Successfully loaded {len(uploaded_files)} user files.")

    # === Step 2: Aggregate Inputs ===
    st.header("Step 2: Aggregating Data")

    # Aggregate Pairwise Matrices
    avg_pairwise = np.mean(pairwise_matrices, axis=0)

    # AHP Weights
    geometric_means = np.prod(avg_pairwise, axis=1) ** (1/avg_pairwise.shape[0])
    ahp_weights = geometric_means / np.sum(geometric_means)

    weights_df = pd.DataFrame({
        'Criteria': criteria,
        'Weight': ahp_weights
    })

    st.subheader("Aggregated AHP Criteria Weights")
    st.dataframe(weights_df.style.format({'Weight': "{:.4f}"}))

    # Aggregate Alternative Ratings (Fuzzy Averaging)
    st.subheader("Aggregated Alternative Ratings (Fuzzy)")

    merged_ratings = {}
    for alt in alternatives:
        merged_ratings[alt] = {}
        for crit in criteria:
            triples = []
            for ratings in alternative_ratings:
                value = ratings.loc[alt, crit]
                if isinstance(value, tuple):
                    triples.append(tuple(float(x) for x in value))
                else:
                    try:
                        # value örnek: (np.float64(0.0), np.float64(0.0), np.float64(1.0))
                        value_str = str(value).replace("np.float64(", "").replace(")", "")
                        numbers = tuple(float(x.strip()) for x in value_str.split(',') if x.strip() != '')
                        triples.append(numbers)
                    except (ValueError, SyntaxError):
                        from collections import defaultdict
                        linguistic_scale = defaultdict(lambda: (0,0,1), {
                            "Very Poor (VP)": (0, 0, 1),
                            "Poor (P)": (0, 1, 3),
                            "Medium Poor (MP)": (1, 3, 5),
                            "Fair (F)": (3, 5, 7),
                            "Medium Good (MG)": (5, 7, 9),
                            "Good (G)": (7, 9, 10),
                            "Very Good (VG)": (9, 10, 10)
                        })
                        triples.append(linguistic_scale[str(value)])
            avg_triple = tuple(np.mean(triples, axis=0))
            merged_ratings[alt][crit] = avg_triple

    df_merged = pd.DataFrame(merged_ratings).T
    st.dataframe(df_merged)

    # === Step 3: Fuzzy TOPSIS Calculation ===
    st.header("Step 3: Fuzzy TOPSIS Results")

    # Normalize
    normalized = {}
    for crit_idx in range(len(criteria)):
        crit_col = [df_merged.loc[alt, criteria[crit_idx]] for alt in alternatives]
        max_upper = max([x[2] for x in crit_col])
        normalized_col = [(x[0]/max_upper, x[1]/max_upper, x[2]/max_upper) for x in crit_col]
        for i, alt in enumerate(alternatives):
            if alt not in normalized:
                normalized[alt] = []
            weighted = tuple(np.array(normalized_col[i]) * ahp_weights[crit_idx])
            normalized[alt].append(weighted)

    # FPIS and FNIS
    fpis = []
    fnis = []
    for i in range(len(criteria)):
        ith_column = [normalized[alt][i] for alt in alternatives]
        fpis.append((max(x[0] for x in ith_column), max(x[1] for x in ith_column), max(x[2] for x in ith_column)))
        fnis.append((min(x[0] for x in ith_column), min(x[1] for x in ith_column), min(x[2] for x in ith_column)))

    def fuzzy_distance(a, b):
        return np.sqrt((1/3) * ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

    results = []
    for alt in alternatives:
        d_pos = sum([fuzzy_distance(normalized[alt][i], fpis[i]) for i in range(len(criteria))])
        d_neg = sum([fuzzy_distance(normalized[alt][i], fnis[i]) for i in range(len(criteria))])
        cci = d_neg / (d_pos + d_neg)
        results.append((alt, round(d_pos, 4), round(d_neg, 4), round(cci, 4)))

    results.sort(key=lambda x: x[3], reverse=True)
    df_results = pd.DataFrame(results, columns=["Alternative", "D+ (FPIS)", "D- (FNIS)", "Closeness Coefficient"])
    df_results['Rank'] = df_results['Closeness Coefficient'].rank(ascending=False, method='min')
    df_results['Rank'] = df_results['Rank'].fillna(0).astype(int)

    st.subheader("Final Ranking")
    st.dataframe(df_results)

    # === Step 4: Download Final Results ===
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save Criteria Weights
        weights_df.to_excel(writer, sheet_name='Criteria Weights', index=False)
        
        # Save Aggregated Ratings
        df_merged.to_excel(writer, sheet_name='Aggregated Ratings')
        
        # Save Fuzzy TOPSIS Results
        df_results.to_excel(writer, sheet_name='Fuzzy TOPSIS Results', index=False)
        
        # Save Consistency Ratio separately
        cr_df = pd.DataFrame({
            'Consistency Ratio (CR)': [CR]
        })
        cr_df.to_excel(writer, sheet_name='Consistency Ratio', index=False)
        
        # Apply Heatmap formatting for Fuzzy TOPSIS Results
        workbook = writer.book
        worksheet = writer.sheets['Fuzzy TOPSIS Results']
    
        # Heatmap for Rank column (E2:E100)
        rank_format = {
            'type': '3_color_scale',
            'min_color': "#63BE7B",   # Green
            'mid_color': "#FFEB84",   # Yellow
            'max_color': "#F8696B",   # Red
        }
        worksheet.conditional_format('E2:E100', rank_format)
    
        # Heatmap for Weight (g) column (F2:F100)
        weight_format = {
            'type': '3_color_scale',
            'min_color': "#63BE7B",
            'mid_color': "#FFEB84",
            'max_color': "#F8696B",
        }
        worksheet.conditional_format('F2:F100', weight_format)
    
    output.seek(0)
    
    st.download_button(
        label="📥 Download Final Report",
        data=output,
        file_name="nvg_final_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

