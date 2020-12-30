
What is it about?
-----------------
- we want functions to be sharable accross projects 
    -> does only work when everyone helps
    -> productivity increases significantly
- modules define interchangable parts/switches of code
    -> replace the same if/else statement at multiple placec in the code by modules
    -> every module defines events, thus, events define interfaces
    -> a module can also sit inbetween the call or the result of each such event
- modules have a second purpose
    -> defining sane defaults wich can be replaced upon request
    -> sotacifar10/sotaimagenet differ (i dont care about the defaults of the modules so much)
- example code

introduction
------------
- start with easy main.py
- show module definition
    - events
    - state
- load the module in main.py
- use defined event
- define another module with same event
- multiple definition of the main event is possible, call order defines execution order
- show miniflask modules: settings, events & info
    - events (module/event regex)
- show run-command for better cli module loading
    - always loaded modules
- fuzzy loading of modules
- fuzzy args for state manipulation
- „replace ifs with modules“

events
------
- unique/optional
    - unique vs non-unique events
    - optional with default altfn
- event["module.id"]
- event.eventname.fns
- event.eventname.modules
- before_/after_ events

state
-----
- types
    - int, float, string
    - bool (with no- flag)
    - list variable
    - variable required by cli
    - list variable required by cli
- all
- `register_defaults` / `register_helpers` / `overwrite_defaults` / `register_globals`
    - exception handling
    - auto-query enabled
- like expressions

advanced topics
---------------
- naming repositories for multi-repository work
- module dependencies
    - relative imports
    - load without event binding
    - load as_id
    - load_as_child
- module structures
    - `set_scope` / `redefine_sope`
    - default module


