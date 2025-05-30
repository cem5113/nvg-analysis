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

    # === Show Aggregated Pairwise Comparison Matrix (Crisp) ===
    st.subheader("Aggregated Pairwise Comparison Matrix (Crisp Values)")
    df_avg_pairwise = pd.DataFrame(avg_pairwise, index=criteria, columns=criteria)
    st.dataframe(df_avg_pairwise.style.format("{:.3f}"))
    
    # === Fuzzy Saaty Scale Mapping Function ===
    def fuzzy_scale(value):
        mapping = {
            1: (1, 1, 1),
            2: (1, 2, 3),
            3: (2, 3, 4),
            4: (3, 4, 5),
            5: (4, 5, 6),
            6: (5, 6, 7),
            7: (6, 7, 8),
            8: (7, 8, 9),
            9: (8, 9, 9),
            1/2: (0.333, 0.5, 1),
            1/3: (0.25, 0.333, 0.5),
            1/4: (0.2, 0.25, 0.333),
            1/5: (0.167, 0.2, 0.25),
            1/6: (0.143, 0.167, 0.2),
            1/7: (0.125, 0.143, 0.167),
            1/8: (0.111, 0.125, 0.143),
            1/9: (0.111, 0.111, 0.125)
        }
        closest = min(mapping.keys(), key=lambda x: abs(x - value))
        return mapping[closest]
    
    # === Convert Crisp Matrix to Fuzzy Matrix ===
    fuzzy_pairwise = np.empty(avg_pairwise.shape, dtype=object)
    
    for i in range(avg_pairwise.shape[0]):
        for j in range(avg_pairwise.shape[1]):
            fuzzy_pairwise[i, j] = fuzzy_scale(avg_pairwise[i, j])
    
    # === Show Fuzzy Aggregated Pairwise Matrix ===
    st.subheader("Fuzzy Aggregated Pairwise Comparison Matrix (Triangular Numbers)")
    df_fuzzy_pairwise = pd.DataFrame(fuzzy_pairwise, index=criteria, columns=criteria)
    st.dataframe(df_fuzzy_pairwise)
        
    # AHP Weights
    geometric_means = np.prod(avg_pairwise, axis=1) ** (1/avg_pairwise.shape[0])
    ahp_weights = geometric_means / np.sum(geometric_means)

    weights_df = pd.DataFrame({
        'Criteria': criteria,
        'Weight': ahp_weights
    })

    # Calculate CR (Consistency Ratio)
    weighted_sum = np.dot(avg_pairwise, ahp_weights)
    lambda_max = np.sum(weighted_sum / ahp_weights) / len(criteria)
    CI = (lambda_max - len(criteria)) / (len(criteria) - 1)
    
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict[len(criteria)]
    CR = CI / RI
    
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
                        val = str(value)
                        
                        if "Very Poor" in val:
                            triples.append((0.0, 0.0, 0.2))
                        elif "Poor" in val and not "Very" in val:
                            triples.append((0.0, 0.2, 0.4))
                        elif "Medium Poor" in val:
                            triples.append((0.2, 0.4, 0.6))
                        elif "Fair" in val:
                            triples.append((0.4, 0.6, 0.8))
                        elif "Medium Good" in val:
                            triples.append((0.6, 0.8, 1.0))
                        elif "Good" in val and not "Very" in val:
                            triples.append((0.8, 1.0, 1.0))
                        elif "Very Good" in val:
                            triples.append((1.0, 1.0, 1.0))
                        else:
                            triples.append((0.0, 0.0, 0.2))  
            avg_triple = tuple(round(x, 1) for x in np.mean(triples, axis=0))
            merged_ratings[alt][crit] = avg_triple

    # Convert all tuples to plain string format (0.0, 0.0, 1.0)
    cleaned_data = {}
    for alt in merged_ratings:
        cleaned_data[alt] = {}
        for crit in merged_ratings[alt]:
            cleaned_data[alt][crit] = tuple(float(x) for x in merged_ratings[alt][crit])
    
    df_merged = pd.DataFrame(cleaned_data).T
    st.dataframe(df_merged)

    # === Step 3: Weighted Fuzzy Decision Matrix ===
    st.subheader("Weighted Fuzzy Decision Matrix")
    
    weighted_fuzzy = {}
    
    for alt in alternatives:
        weighted_fuzzy[alt] = {}
        for idx, crit in enumerate(criteria):
            triple = df_merged.loc[alt, crit]
            weight = ahp_weights[idx]
            if isinstance(triple, tuple) or isinstance(triple, list):
                weighted_triple = tuple(round(float(x) * weight, 3) for x in triple)
            else:
                weighted_triple = round(triple * weight, 3)
            weighted_fuzzy[alt][crit] = f"({weighted_triple[0]}, {weighted_triple[1]}, {weighted_triple[2]})"
    
    df_weighted_fuzzy = pd.DataFrame(weighted_fuzzy).T
    st.dataframe(df_weighted_fuzzy)

    # === Step 4: Fuzzy TOPSIS Calculation ===
    st.header("Step 3: Fuzzy TOPSIS Results")

    # Normalize
    normalized = {}
    for crit_idx in range(len(criteria)):
        crit_col = [df_merged.loc[alt, criteria[crit_idx]] for alt in alternatives]
        crit_col = [ast.literal_eval(x) if isinstance(x, str) else x for x in crit_col]  
        max_upper = max([x[2] for x in crit_col])
        normalized_col = []
        for x in crit_col:
            try:
                if isinstance(x, str):
                    x_tuple = ast.literal_eval(x)
                elif isinstance(x, (tuple, list)):
                    x_tuple = x
                else:
                    x_tuple = (0.0, 0.0, 0.0) 
                
                # Şimdi normalle
                normalized_col.append((
                    float(x_tuple[0])/max_upper,
                    float(x_tuple[1])/max_upper,
                    float(x_tuple[2])/max_upper
                ))
            except Exception as e:
                # Hatalı veri olursa güvenli default
                normalized_col.append((0.0, 0.0, 0.0))
    
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

    nvg_attributes = {
        "NVG A": {"Weight (g)": 592, "Phosphor Screen": "Green"},
        "NVG B": {"Weight (g)": 506, "Phosphor Screen": "Green"},
        "NVG C": {"Weight (g)": 424, "Phosphor Screen": "White"},
        "NVG D": {"Weight (g)": 525, "Phosphor Screen": "Green"},
        "NVG E": {"Weight (g)": 650, "Phosphor Screen": "White"},
        "NVG F": {"Weight (g)": 560, "Phosphor Screen": "Green"},
        "NVG G": {"Weight (g)": 600, "Phosphor Screen": "White"}
    }
    
    # Add Weight (g) and Phosphor Screen columns
    df_results["Weight (g)"] = df_results["Alternative"].map(lambda x: nvg_attributes[x]["Weight (g)"])
    df_results["Phosphor Screen"] = df_results["Alternative"].map(lambda x: nvg_attributes[x]["Phosphor Screen"])
    
    st.subheader("Final Ranking")
    st.dataframe(df_results)

    # === Step 5: Download Final Results ===
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save Aggregated Pairwise Matrix to Excel
        df_avg_pairwise.to_excel(writer, sheet_name='Agg Pairwise Matrix')

        # Save Aggregated Crisp Pairwise Matrix
        df_avg_pairwise.to_excel(writer, sheet_name='Agg Crisp Pairwise Matrix')
    
        # Save Fuzzy Aggregated Pairwise Matrix
        df_fuzzy_pairwise.to_excel(writer, sheet_name='Agg Fuzzy Pairwise Matrix')

        # Save Criteria Weights
        weights_df.to_excel(writer, sheet_name='Criteria Weights', index=False)
        
        # Save Aggregated Ratings
        df_merged.to_excel(writer, sheet_name='Aggregated Ratings')
        
        # Save Weighted Fuzzy Ratings
        df_weighted_fuzzy.to_excel(writer, sheet_name='Weighted Fuzzy Ratings')
        
        # Save Fuzzy TOPSIS Results
        df_results.to_excel(writer, sheet_name='Fuzzy TOPSIS Results', index=False)
        
        # Save Consistency Indicator 
        ci_df = pd.DataFrame({
            'Consistency Indicator (CI)': [CI]
        })
        ci_df.to_excel(writer, sheet_name='Consistency Indicator', index=False)
        
        # Save Consistency Ratio 
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

