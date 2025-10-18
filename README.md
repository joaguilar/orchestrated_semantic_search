# orchestrated_semantic_search
Semantic Search demo using an orchestration framework (Semantic Kernel) and Promptflow

## VS Code Extensions:

- Azure Resources
- Semantic Kernel
- Prompt Flow for VS Code

## Reference Docs:

[pf](https://microsoft.github.io/promptflow/reference/pf-command-reference.html)




# Steps to create the Application:

```
pf flow init --flow my-copilot --type chat
```

Create the Azure OpenAI connection:

```
pf connection create --file azure_openai.yaml --name open_ai_connection
```

**Note: in the flow.dag.yaml file, remove the following lines for gpt-#-nano models that don't support temperature or max_tokens:**

```
    max_tokens: "256"
    temperature: "0.7"
```

Then test out the "simple" copilot:

```
❯ pf flow test --flow my-copilot --interactive
Prompt flow service has started...
=================================
Welcome to chat flow, my_copilot.
Press Enter to send your message.
You can quit with ctrl+C.
=================================
User: Hello
Bot: Hi there! How can I help today? I can assist with a wide range of things—answer questions, explain concepts, help with writing or editing, brainstorm ideas, code or debug, plan a trip, summarize articles, and more. What would you like to do?
User: 
```

