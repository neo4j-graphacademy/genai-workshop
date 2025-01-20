def test_llm_prompt(test_helpers, monkeypatch):

    output = test_helpers.run_module(monkeypatch, "llm_prompt")

    assert output > ""

def test_llm_chain_string(test_helpers, monkeypatch):

    assert test_helpers.run_module(monkeypatch, "llm_chain_string") > ""

def test_llm_chain_json(test_helpers, monkeypatch):

    assert test_helpers.run_module(monkeypatch, "llm_chain_json") > ""

def test_chat_model_context(test_helpers, monkeypatch):

    assert test_helpers.run_module(monkeypatch, "chat_model_context") > ""

def test_chat_model_memory_neo4j(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "chat_model_memory",
        ["Whats happening at Fistral?", "exit"]
    )
    
    # Test a response was received from the agent
    # There is a output which looks like Session ID: #####\n[response from LLM]\n
    assert len(output.split("\n")) == 2

def test_chat_agent(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "chat_agent",
        ["Find a movie about the meaning of life", "exit"]
    )
    
    # Test a response was received from the agent
    # There is a output which looks like Session ID: #####\n[response from LLM]\n
    assert len(output.split("\n")) == 2

def test_chat_agent_retriever(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "chat_agent_retriever",
        [
            "Find a movie with a plot about a mission to the moon that goes wrong",
            "exit"
        ]
    )
    
    assert len(output.split("\n")) >= 2

def test_chat_agent_trailer(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "chat_agent_trailer",
        ["Find the movie trailer for the Matrix.", "exit"]
    )
    
    assert len(output.split("\n")) >= 2