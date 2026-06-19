import React from 'react';
import { Rocket, Loader, RefreshCw, Target, Zap } from 'lucide-react';

const ChatTab = ({
  autoIndexFromNASA,
  autoIndexing,
  chatMessages,
  kbStatus,
  setChatInput,
  chatInput,
  handleChatSubmit,
  chatLoading,
  useAgent,
  setUseAgent
}) => {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl border border-slate-200 shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="bg-white/20 backdrop-blur rounded-xl p-3">
                <Rocket className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-2xl font-bold">NEO AI Assistant</h3>
                <p className="text-indigo-100 text-sm">Ask anything about Near-Earth Objects</p>
              </div>
            </div>
            <div className="flex items-center gap-3 bg-white/10 px-4 py-2 rounded-xl backdrop-blur-sm border border-white/20">
              <span className="text-sm font-medium text-white">LangGraph Agent</span>
              <button
                onClick={() => setUseAgent(!useAgent)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:ring-offset-2 ${
                  useAgent ? 'bg-indigo-400' : 'bg-slate-400/50'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    useAgent ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
          <button
            onClick={autoIndexFromNASA}
            disabled={autoIndexing}
            className={`mt-4 px-5 py-3 bg-white text-indigo-600 rounded-xl font-semibold flex items-center gap-2 hover:bg-indigo-50 transition-all ${
              autoIndexing ? 'opacity-70 cursor-not-allowed' : ''
            }`}
          >
            {autoIndexing ? (
              <>
                <Loader className="animate-spin" size={18} />
                Indexing NASA Data...
              </>
            ) : (
              <>
                <RefreshCw size={18} />
                Auto-Index from NASA
              </>
            )}
          </button>
        </div>

        {/* Messages Area */}
        <div className="h-96 overflow-y-auto p-6 bg-gradient-to-b from-slate-50 to-white">
          {chatMessages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center">
              <div className="max-w-md">
                <div className="bg-indigo-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6">
                  <Target className="w-10 h-10 text-indigo-600" />
                </div>
                <h4 className="text-xl font-bold text-slate-800 mb-3">
                  {kbStatus?.document_count === 0
                    ? "Knowledge base is empty"
                    : "How can I help you today?"}
                </h4>
                <p className="text-slate-600 mb-6">
                  {kbStatus?.document_count === 0
                    ? "Click the button above to load asteroid data first."
                    : "Ask about the most dangerous NEOs, risk analysis, impact zones, or any specific asteroid."}
                </p>

                {kbStatus?.document_count > 0 && (
                  <div className="grid grid-cols-1 gap-3 max-w-lg mx-auto">
                    {[
                      "What are the most dangerous asteroids approaching Earth?",
                      "Which NEOs require immediate follow-up?",
                      "Show me the highest risk objects this month",
                      "Compare the top 3 critical risk asteroids"
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => setChatInput(q)}
                        className="text-left p-4 bg-white border border-slate-200 rounded-xl hover:border-indigo-400 hover:shadow-md transition-all text-sm text-slate-700"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl rounded-2xl px-5 py-4 shadow-sm ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white border border-slate-200 text-slate-800'
                    }`}
                  >
                    {/* Render assistant message with proper formatting */}
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm max-w-none text-slate-800">
                        {/* Replace markdown-like formatting with proper HTML */}
                        {msg.content
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-700 font-bold">$1</strong>')
                          .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
                          .replace(/^\d+\.\s/gm, (match) => `<strong class="text-indigo-700">${match}</strong>`)
                          .split('\n')
                          .map((line, i) => {
                            if (line.trim() === '') return <br key={i} />;
                            if (line.startsWith('•') || line.startsWith('- '))
                              return (
                                <div key={i} className="flex items-start gap-3 mt-2">
                                  <span className="text-indigo-600 mt-1.5">●</span>
                                  <span dangerouslySetInnerHTML={{ __html: line.slice(2) }} />
                                </div>
                              );
                            return (
                              <p key={i} className="mb-3 last:mb-0" dangerouslySetInnerHTML={{ __html: line }} />
                            );
                          })}
                      </div>
                    ) : (
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                    )}

                    {/* Sources */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-slate-200">
                        <p className="text-xs font-semibold text-slate-600 mb-3">
                          📊 Sources ({msg.sources.length})
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {msg.sources.slice(0, 6).map((source, i) => (
                            <div
                              key={i}
                              className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-3 border border-indigo-100"
                            >
                              <p className="font-bold text-indigo-700 text-xs truncate">
                                {source.name || `NEO ${source.neo_id}`}
                              </p>
                              <div className="flex items-center gap-2 mt-1 text-xs text-slate-600">
                                <span>Risk: {source.risk_score?.toFixed(2)}</span>
                                <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded-full font-bold">
                                  {source.risk_category}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm flex items-center gap-3">
                    <Loader className="animate-spin text-indigo-600" size={20} />
                    <span className="text-slate-600">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-5 bg-white border-t border-slate-200">
          <div className="flex gap-3">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSubmit()}
              placeholder={
                kbStatus?.document_count === 0
                  ? "Index data first to enable AI assistant..."
                  : "Ask about asteroids, risks, impact zones..."
              }
              className="flex-1 px-5 py-4 bg-slate-50 border border-slate-300 rounded-xl text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent placeholder-slate-400"
              disabled={chatLoading || kbStatus?.document_count === 0}
            />
            <button
              onClick={handleChatSubmit}
              disabled={chatLoading || !chatInput.trim() || kbStatus?.document_count === 0}
              className={`px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold flex items-center gap-3 transition-all shadow-lg ${
                chatLoading || !chatInput.trim() || kbStatus?.document_count === 0
                  ? 'opacity-50 cursor-not-allowed'
                  : 'hover:from-indigo-700 hover:to-purple-700 hover:shadow-xl'
              }`}
            >
              {chatLoading ? (
                <Loader className="animate-spin" size={22} />
              ) : (
                <Zap size={22} />
              )}
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;
