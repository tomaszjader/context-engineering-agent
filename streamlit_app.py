import streamlit as st
import main
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Context Engineering Agent", page_icon="🤖", layout="wide")

st.title("Context Engineering Agent UI")
st.markdown("Visualizing the execution of the main context engineering agent.")

goal = st.text_input(
    "Enter your research goal:",
    value="Search for popular free AI courses and recommend 3 best focused on building RAG systems with LangChain.",
)

if st.button("Run Agent", type="primary"):
    telemetry = main.RunTelemetry()
    
    with st.status("Planning steps...", expanded=True) as status:
        steps = main.create_plan(goal, telemetry)
        if not steps:
            status.update(label="No executable steps were produced by the planner.", state="error")
            st.stop()
        else:
            status.update(label=f"Planner created {len(steps)} steps.", state="complete")
    
    st.subheader("Planned Steps")
    for i, step in enumerate(steps, start=1):
        st.write(f"**Step {i}:** {step}")
        
    summary_lines = ["- Task started."]
    
    st.write("---")
    st.subheader("Execution")
    
    for i, step in enumerate(steps, start=1):
        with st.expander(f"Step {i}/{len(steps)}: {step}", expanded=True):
            with st.spinner("Executing step..."):
                current_summary = "\n".join(summary_lines)
                
                result = main.execute_step(step, current_summary, telemetry)
                
                st.markdown("**Result:**")
                st.info(result)
                
                summary_lines = main.update_state_summary(summary_lines, step, result)
                main.save_to_memory(result)
                
    st.write("---")
    st.subheader("Finalizing")
    
    with st.spinner("Generating final answer..."):
        final_prompt = (
            "Create the final answer from execution summary.\n"
            "Requirements: concrete recommendations, no fluff, max 180 words.\n\n"
            f"Goal: {goal}\n"
            f"Execution summary:\n{chr(10).join(summary_lines)}"
        )
        
        final_answer = main._invoke_with_telemetry(
            main.final_llm,
            [HumanMessage(content=final_prompt)],
            telemetry,
            "final",
        ).content
        
    st.success("Agent run completed successfully!")
    st.markdown("### Final Answer")
    st.write(final_answer)
    
    st.write("---")
    st.subheader("Telemetry & Cost Report")
    st.write(f"**Estimated Cost:** ${telemetry.estimate_cost_usd():.6f} (excludes embeddings/search, approx pricing)")
    
    if telemetry.by_stage:
        telemetry_data = []
        for stage, usage in telemetry.by_stage.items():
            telemetry_data.append({
                "Stage": stage,
                "Input Tokens": usage.input_tokens,
                "Output Tokens": usage.output_tokens,
                "Total Tokens": usage.total_tokens
            })
        st.table(telemetry_data)
