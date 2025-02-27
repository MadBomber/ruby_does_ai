# Agent99

**Under Development**  Initial release has no AI components - its just a generic client-server / request-response micro-services system using a peer-to-peer messaging broker and a centralized agent registry.  To keep up with the version changes review [The Changelog](./CHANGELOG.md) file.

v0.0.4 has a [breaking_change.](docs/breaking_change_v0.0.4.md)

Agent99 is a Ruby-based framework for building and managing AI agents in a distributed system. It provides a robust foundation for creating intelligent agents that can communicate, discover each other, and perform various tasks.

## Hype!

🌟 **Introducing Agent99**: Your Friendly Ruby Sidekick for Building Intelligent Software Agents! 🌟

Do you ever miss the charm and wit of Barbara Feldon's iconic character from *Get Smart*? Just as Agent 99 was always ready to tackle mischief and save the day, **Agent99** is here to help you create your very own software agents that can communicate, collaborate, and conquer tasks in a smart and efficient way!

### Why Choose Agent99?

🔍 **Independent Agents, Unified Communication**: Say goodbye to complex service orchestration! Agent99 empowers you to build micro-services that communicate seamlessly through a robust peer-to-peer messaging network. Each agent operates independently while working together like a well-oiled machine!

🤖 **Smart Automation for the Modern Age**: In the era of AI, why settle for less? With Agent99, you can develop software agents that respond intelligently to incoming requests—just like a secret agent reacting to their next mission! Deploy your own digital spies to gather data, complete tasks, and more.

📦 **Quick Setup & Easy Integration**: Just like Agent 99’s quick thinking, our library is designed for rapid development. Get up and running in no time, whether you’re building a small project or an enterprise-level solution!


### Features that Make Agent99 a Must-Have:

- 🌐 **Peer-to-Peer Messaging**: Ensure that your agents can communicate effortlessly, just like Agent 99 and Max.
- 🚀 **Agile Development**: Design and deploy agents at lightning speed, giving you the freedom to innovate.
- 🔒 **Secure Communication**: Built with security in mind, because even our agents deserve to keep their secrets safe.
- 📊 **Scalability**: Expand your network of agents as your needs grow—no mission is too big!

### Get Smart! Join the Revolution!

Whether you’re a seasoned Ruby developer or just getting started, Agent99 provides the tools you need to build your very own quirky, intelligent agents. Just like Agent 99, your agents will be clever, adaptable, and ready to tackle any challenge!

### How to Get Started:

1. **Install**: Simply add Agent99 to your Gemfile and run `bundle install`.
2. **Create an Agent**: Use our simple API to define and deploy your first agent.
3. **Watch Them Work**: Sit back and relax as your agents spring into action, communicating with one another to accomplish tasks that would impress even the chief of CONTROL!

### Spread the Word!

**Join the Agent99 community** on GitHub and share your experiences, tips, and feedback. Let’s build a world of smart agents together! And remember, just like Agent 99 always had Max’s back, we’ve got yours during your development journey.

🕵️‍♂️ **Become an Agent!** Dive into the world of Agent99 today: [GitHub Repository](#) 📖

**Agent99** – Because when it comes to building software agents, it's all about being smart!

... end of the Hype; now back to normal.

## Features

- Agent Lifecycle Management: Easy setup and teardown of agents
- Message Processing: Handle requests, responses, and control messages
- Agent Discovery: Find other agents based on capabilities
- Flexible Communication: Support for both AMQP and NATS messaging systems
- Registry Integration: Register and discover agents through a central registry
- Error Handling and Logging: Built-in error management and logging capabilities
- Control Actions: Pause, resume, update configuration, and request status of agents
- Dynamic Agent Loading: Support for runtime loading and deployment of new agents
- Multi-Agent Processing: Run multiple agents within the same process using thread isolation

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'agent99'
```

And then execute:

```
$ bundle install
```

Or install it yourself as:

```
$ gem install agent99
```

## Usage

Here's a basic example of how to create an AI agent:

```ruby
require 'agent99'

class GreeterRequest < SimpleJsonSchemaBuilder::Base
  object do
    object :header, schema: Agent99::HeaderSchema
    string :name, required: true, examples: ["World"]
  end
end

class GreeterAgent < Agent99::Base
  def info
    {
      name:             self.class.to_s,
      type:             :server,
      capabilities:     ['greeter', 'hello_world'],
      request_schema:   GreeterRequest.schema,
      # Uncomment and define these schemas as needed:
      # response_schema:  {}, # Agent99::RESPONSE.schema
      # control_schema:   {}, # Agent99::CONTROL.schema
      # error_schema:     {}, # Agent99::ERROR.schema
    }
  end

  def process_request(payload)
    name      = payload.dig(:name)
    response  = { result: "Hello, #{name}!" }
    send_response(response)
  end
end

# Create and run the agent
agent = GreeterAgent.new
agent.run
```

## Configuration

The framework can be configured through environment variables:

- `AGENT99_REGISTRY_URL`: URL of the agent registry service (default: 'http://localhost:4567')

Depending on which messaging client you are using, additional environment variables may be used.

TODO: show envars for AMQP via Bunny
TODO: show envars for NATS via nats0server

See the examples folder for a default registry service implementation.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/MadBomber/agent99.

## Short-term Roadmap

- In the example registry, replace the Array(Hash) datastore with sqlite3 with a vector table to support discovery using semantic search.
- Treat the agent like a Tool w/r/t RAG for prompts.
- Add AgentRequest schema to agent's info in the registry.
- Add AgentResponse schema to define the `result` element in the response JSON payload

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
