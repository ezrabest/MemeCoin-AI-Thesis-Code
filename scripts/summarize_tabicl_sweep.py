import json
import glob
import pandas as pd
from pathlib import Path

def summarize_reports():
    report_files = glob.glob("data/training/policy_backtests/tabicl_v2_report_*.json")
    
    if not report_files:
        print("No reports found!")
        return

    results = []
    for file in report_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            strat = data.get("context_strategy", "unknown")
            ctx_size = data.get("context_size_used", 0)
            
            # תיקון: משיכת הנתונים משני המיקומים האפשריים ב-JSON
            val_data = data.get("validation") or data.get("tabicl_metrics", {}).get("validation") or {}
            test_data = data.get("test") or data.get("tabicl_metrics", {}).get("test") or {}
            
            val_prec1 = val_data.get("precision_at_top_1_percent", 0)
            test_prec1 = test_data.get("precision_at_top_1_percent", 0)
            
            val_ret1 = val_data.get("total_target_return_4h_top_1_percent", 0)
            test_ret1 = test_data.get("total_target_return_4h_top_1_percent", 0)

            results.append({
                "Strategy": strat,
                "Context Size": ctx_size,
                "Val Prec@1%": f"{val_prec1:.1%}" if val_prec1 else "0.0%",
                "Test Prec@1%": f"{test_prec1:.1%}" if test_prec1 else "0.0%",
                "Val Return": f"{val_ret1:+.2f}" if val_ret1 else "+0.00",
                "Test Return": f"{test_ret1:+.2f}" if test_ret1 else "+0.00"
            })

    df = pd.DataFrame(results)
    # מיון לפי ההצלחה ב-Test (כדי לראות מי מכליל הכי טוב קדימה) ואז לפי Validation
    df = df.sort_values(by=["Test Prec@1%", "Val Prec@1%"], ascending=[False, False]).reset_index(drop=True)
    
    # הדפסה לטרמינל
    print("\n" + "="*70)
    print("🏆 TabICLv2 Strategy Sweep Results")
    print("="*70)
    print(df.to_markdown())
    print("="*70 + "\n")

    # שמירה לקבצים כדי לעקוף את מגבלת הקונסולה!
    out_csv = "data/training/policy_backtests/FINAL_SWEEP_SUMMARY.csv"
    out_md = "data/training/policy_backtests/FINAL_SWEEP_SUMMARY.md"
    
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df.to_markdown())
    
    print(f"✅ Summary CSV saved to: {out_csv}")
    print(f"✅ Summary Markdown saved to: {out_md}")

if __name__ == "__main__":
    summarize_reports()